"""
training/trainer.py

Training loop for PI-SLM using time-series ΔG data.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders, HBD2Dataset
from model.transformer import PISLM
from physics.constraints import PhysicsConstraints


# ─────────────────────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_data_loss(predictions, batch, device):
    """
    Compute data loss — ΔG regression only.
    Goal is accurate ΔG prediction for sequence ranking.
    """
    y_dg     = batch['delta_g'].to(device)
    y_rmsd   = batch['rmsd'].to(device)
    y_hbonds = batch['h_bonds'].to(device)
    has_rmsd = batch['has_rmsd'].to(device)
    has_hb   = batch['has_hbonds'].to(device)

    # Primary: ΔG regression on all records
    loss_dg = nn.MSELoss()(predictions['delta_g'], y_dg)

    # Secondary: RMSD where available
    if has_rmsd.sum() > 0:
        loss_rmsd = nn.MSELoss()(
            predictions['rmsd'][has_rmsd],
            y_rmsd[has_rmsd]
        )
    else:
        loss_rmsd = torch.tensor(0.0, device=device)

    # Secondary: H-bonds where available
    if has_hb.sum() > 0:
        loss_hb = nn.MSELoss()(
            predictions['h_bonds'][has_hb],
            y_hbonds[has_hb]
        )
    else:
        loss_hb = torch.tensor(0.0, device=device)

    total = (
        1.0 * loss_dg   +
        0.3 * loss_rmsd +
        0.2 * loss_hb
    )

    return total, {
        'dg':     loss_dg.item(),
        'rmsd':   loss_rmsd.item(),
        'hbonds': loss_hb.item(),
    }


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_dg_true     = []
    all_dg_pred     = []
    phys_violations = 0
    total_records   = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        physchem  = batch['physchem'].to(device)
        time_norm = batch['time_norm'].to(device)

        preds = model(input_ids, physchem, time_norm)

        pred_dg = (preds['delta_g'].cpu().numpy()
                   * HBD2Dataset.DG_STD + HBD2Dataset.DG_MEAN)
        true_dg = (batch['delta_g'].numpy()
                   * HBD2Dataset.DG_STD + HBD2Dataset.DG_MEAN)

        all_dg_true.extend(true_dg.tolist())
        all_dg_pred.extend(pred_dg.tolist())

        # Only count violations on true binders
        true_binder_mask = true_dg < 0
        violations = np.sum(pred_dg[true_binder_mask] > 0)
        phys_violations += violations
        total_records   += true_binder_mask.sum()

    dg_true = np.array(all_dg_true)
    dg_pred = np.array(all_dg_pred)

    dg_rmse   = float(np.sqrt(np.mean((dg_true - dg_pred) ** 2)))
    ss_res    = np.sum((dg_true - dg_pred) ** 2)
    ss_tot    = np.sum((dg_true - dg_true.mean()) ** 2)
    r2        = float(1 - ss_res / (ss_tot + 1e-8))
    viol_rate = (phys_violations / max(total_records, 1)) * 100

    return {
        'dg_rmse':        dg_rmse,
        'r2':             r2,
        'violation_rate': viol_rate,
    }


# ─────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────

def train(
    config_path     = 'configs/training.yaml',
    model_config    = 'configs/model.yaml',
    physics_config  = 'configs/physics.yaml',
    pretrained_path = None,
    excel_path      = 'data/raw/consolidated_cleaned.xlsx',
):
    # ── Load config ───────────────────────────────────────────
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    tcfg     = cfg['training']['finetune']
    device   = tcfg['device']
    epochs   = tcfg['epochs']
    lr       = tcfg['learning_rate']
    patience = tcfg['patience']
    bs       = tcfg['batch_size']
    seed     = cfg['training']['data']['random_seed']

    print("=" * 70)
    print("PI-SLM TRAINING — TIME SERIES MODE")
    print("=" * 70)
    print(f"Device  : {device}")
    print(f"Epochs  : {epochs}")
    print(f"LR      : {lr}")
    print(f"Patience: {patience}")
    print(f"Batch   : {bs}")
    print(f"Data    : {excel_path}")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        excel_path=excel_path,
        batch_size=bs,
        random_seed=seed,
    )

    # ── Model ─────────────────────────────────────────────────
    model = PISLM(config_path=model_config).to(device)

    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt       = torch.load(pretrained_path, map_location=device)
        state_dict = {
            k: v for k, v in ckpt['model_state_dict'].items()
            if 'pos_encoding.pe' not in k
        }
        model.load_state_dict(state_dict, strict=False)
        print(f"\nLoaded pretrained weights from: {pretrained_path}")
        print(f"  (positional encoding reinitialized for seq_len=13)")
    else:
        print("\nTraining from random initialization")

    # ── Physics constraints ───────────────────────────────────
    physics = PhysicsConstraints(config_path=physics_config)

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # ── Checkpointing ─────────────────────────────────────────
    ckpt_dir  = Path(cfg.get('checkpoints', {}).get(
        'save_dir', 'results/checkpoints'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / cfg.get('checkpoints', {}).get(
        'best_model_name', 'best_model.pt')

    # ── Training loop ─────────────────────────────────────────
    best_val_rmse = float('inf')
    patience_ctr  = 0
    training_log  = []  # records metrics every 5 epochs

    print("\n" + "-" * 70)
    print(f"{'Epoch':>6} | {'DataLoss':>9} | {'DG_RMSE':>8} | "
          f"{'R2':>6} | {'Viol%':>6} | {'PhysW':>7}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            physchem  = batch['physchem'].to(device)
            time_norm = batch['time_norm'].to(device)

            # Forward pass
            predictions = model(input_ids, physchem, time_norm)

            # Data loss
            data_loss, loss_parts = compute_data_loss(
                predictions, batch, device
            )

            # Physics loss — only on true binders
            binder_mask = (batch['raw_delta_g'].to(device) < 0)
            phys_result = physics(predictions, binder_mask, epoch)
            phys_loss   = phys_result['total']

            total_loss = data_loss + phys_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg['training']['grad_clip']
            )
            optimizer.step()
            epoch_losses.append(total_loss.item())

        scheduler.step()

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            metrics   = evaluate(model, val_loader, device)
            mean_loss = np.mean(epoch_losses)

            print(
                f"{epoch:>6} | "
                f"{mean_loss:>9.4f} | "
                f"{metrics['dg_rmse']:>8.3f} | "
                f"{metrics['r2']:>6.3f} | "
                f"{metrics['violation_rate']:>5.1f}% | "
                f"{phys_result['weight']:>7.4f}"
            )

            # Save to training log
            training_log.append({
                'epoch':     epoch,
                'data_loss': mean_loss,
                'dg_rmse':   metrics['dg_rmse'],
                'r2':        metrics['r2'],
                'viol_rate': metrics['violation_rate'],
                'phys_weight': phys_result['weight'],
            })

            if metrics['dg_rmse'] < best_val_rmse:
                best_val_rmse = metrics['dg_rmse']
                patience_ctr  = 0
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': model.state_dict(),
                    'val_dg_rmse':      best_val_rmse,
                    'val_r2':           metrics['r2'],
                    'config':           model_config,
                }, best_path)
                print(f"         ✓ Saved best model "
                      f"(RMSE={best_val_rmse:.3f})")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    # ── Save training log ─────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    log_df = pd.DataFrame(training_log)
    log_df.to_csv('results/training_log.csv', index=False)
    print(f"\nTraining log saved to results/training_log.csv")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Best val ΔG RMSE : {best_val_rmse:.3f} kcal/mol")
    print(f"  Model saved to   : {best_path}")
    print("=" * 70)

    # Final test evaluation
    print("\nFinal evaluation on test set:")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device)
    print(f"  Test ΔG RMSE      : {test_metrics['dg_rmse']:.3f} kcal/mol")
    print(f"  Test R²           : {test_metrics['r2']:.3f}")
    print(f"  Physics Violations: {test_metrics['violation_rate']:.1f}%")

    return test_metrics


# ─────────────────────────────────────────────────────────────
# RUN STANDALONE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(
        config_path     = 'configs/training.yaml',
        model_config    = 'configs/model.yaml',
        physics_config  = 'configs/physics.yaml',
        pretrained_path = 'results/checkpoints/pretrained_model.pt',
        excel_path      = 'data/raw/consolidated_cleaned.xlsx',
    )