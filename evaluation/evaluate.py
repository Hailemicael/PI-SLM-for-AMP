"""
evaluation/evaluate.py

Evaluation and sequence generation for PI-SLM.

Two jobs:
    1. Evaluate trained model on test set
    2. Generate new candidate sequences ranked by predicted ΔG
"""

import torch
import numpy as np
import pandas as pd
import yaml
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import (
    get_dataloaders, HBD2Dataset,
    encode_sequence, encode_physchem, VALID_AAS
)
from model.transformer import PISLM

ALL_AAS = list('ACDEFGHIKLMNPQRSTVWY')
SEQ_LEN = 13


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, model_config, device):
    model = PISLM(config_path=model_config).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Trained for {ckpt['epoch']} epochs")
    print(f"  Best val ΔG RMSE: {ckpt['val_dg_rmse']:.3f} kcal/mol")
    return model


# ─────────────────────────────────────────────────────────────
# EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_set(model, test_loader, device):
    model.eval()

    results = []

    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        physchem  = batch['physchem'].to(device)
        time_norm = batch['time_norm'].to(device)

        preds = model(input_ids, physchem, time_norm)

        pred_dg = (preds['delta_g'].cpu().numpy()
                   * HBD2Dataset.DG_STD + HBD2Dataset.DG_MEAN)
        true_dg = (batch['delta_g'].numpy()
                   * HBD2Dataset.DG_STD + HBD2Dataset.DG_MEAN)

        pred_rmsd = (preds['rmsd'].cpu().numpy()
                     * HBD2Dataset.RMSD_STD + HBD2Dataset.RMSD_MEAN)

        for i in range(len(true_dg)):
            results.append({
                'sequence':  batch['sequence'][i],
                'true_dg':   true_dg[i],
                'pred_dg':   pred_dg[i],
                'pred_rmsd': pred_rmsd[i],
                'time_norm': batch['time_norm'][i].item(),
            })

    df = pd.DataFrame(results)

    dg_true = df['true_dg'].values
    dg_pred = df['pred_dg'].values

    rmse = float(np.sqrt(np.mean((dg_true - dg_pred) ** 2)))
    mae  = float(np.mean(np.abs(dg_true - dg_pred)))
    ss_res = np.sum((dg_true - dg_pred) ** 2)
    ss_tot = np.sum((dg_true - dg_true.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-8))

    binder_mask = dg_true < 0
    violations  = np.sum(dg_pred[binder_mask] > 0)
    viol_rate   = violations / max(binder_mask.sum(), 1) * 100

    metrics = {
        'rmse':       rmse,
        'mae':        mae,
        'r2':         r2,
        'viol_rate':  viol_rate,
        'n_test':     len(df),
        'n_binders':  binder_mask.sum(),
    }

    return metrics, df


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_predictions(df, metrics, save_dir='results/figures'):
    """
    Generate all publication-quality figures.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Set style
    plt.rcParams.update({
        'font.size':        12,
        'font.family':      'serif',
        'axes.linewidth':   1.2,
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'figure.dpi':       150,
    })

    # ── Figure 1: Predicted vs True ΔG ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    dg_true = df['true_dg'].values
    dg_pred = df['pred_dg'].values

    ax.scatter(
        dg_true, dg_pred,
        alpha=0.3, s=8, color='steelblue', label='Time points'
    )

    # Perfect prediction line
    lim = [min(dg_true.min(), dg_pred.min()) - 2,
           max(dg_true.max(), dg_pred.max()) + 2]
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlabel('True ΔG (kcal/mol)', fontsize=13)
    ax.set_ylabel('Predicted ΔG (kcal/mol)', fontsize=13)
    ax.set_title('PI-SLM: Predicted vs True Binding Free Energy', fontsize=14)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # Annotate metrics
    ax.text(
        0.05, 0.95,
        f'RMSE = {metrics["rmse"]:.2f} kcal/mol\n'
        f'R² = {metrics["r2"]:.3f}\n'
        f'MAE = {metrics["mae"]:.2f} kcal/mol\n'
        f'Physics violations = {metrics["viol_rate"]:.1f}%',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/predicted_vs_true_dg.png', dpi=300)
    plt.close()
    print(f"  Saved: predicted_vs_true_dg.png")

    # ── Figure 2: Error distribution ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    errors = dg_pred - dg_true
    ax.hist(errors, bins=60, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero error')
    ax.axvline(errors.mean(), color='orange', linestyle='--',
               linewidth=1.5, label=f'Mean error = {errors.mean():.2f}')

    ax.set_xlabel('Prediction Error (kcal/mol)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Distribution of Prediction Errors', fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distribution.png', dpi=300)
    plt.close()
    print(f"  Saved: error_distribution.png")




def plot_training_history(log_path='results/training_log.csv',
                          save_dir='results/figures'):
    """
    Plot training curves — loss, R², physics violations over epochs.
    Reads from training log CSV if available.
    """
    if not Path(log_path).exists():
        print(f"  No training log found at {log_path} — skipping curves")
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss curve
    axes[0].plot(df['epoch'], df['data_loss'],
                 color='steelblue', linewidth=1.5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Data Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=13)
    axes[0].grid(alpha=0.3)

    # R² curve
    axes[1].plot(df['epoch'], df['r2'],
                 color='green', linewidth=1.5)
    axes[1].axhline(0.717, color='orange', linestyle='--',
                    linewidth=1.2, label='CNN baseline (0.717)')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R²', fontsize=12)
    axes[1].set_title('R² Score over Training', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.0)

    # Physics violations curve
    axes[2].plot(df['epoch'], df['viol_rate'],
                 color='red', linewidth=1.5)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Physics Violations (%)', fontsize=12)
    axes[2].set_title('Physics Violations over Training', fontsize=13)
    axes[2].grid(alpha=0.3)

    plt.suptitle('PI-SLM Training Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300)
    plt.close()
    print(f"  Saved: training_curves.png")


# ─────────────────────────────────────────────────────────────
# GENERATE NEW SEQUENCES
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sequences(model, device, n_generate=30000, top_k=30,
                        dg_threshold=-40.0, excel_path=None):
    """
    Generate new candidate AMP sequences.
    Rank by predicted ΔG — most negative = strongest binding.
    """
    print(f"\nGenerating {n_generate:,} random sequences...")
    print(f"Target: predicted ΔG < {dg_threshold} kcal/mol")

    model.eval()

    # Load known sequences to exclude
    if excel_path:
        from data.dataset import build_dataset
        all_data = build_dataset(excel_path)
        known = set(d['sequence'] for d in all_data)
    else:
        known = set()
    print(f"Excluding {len(known)} known sequences")

    all_results = []
    batch_size  = 512
    generated   = 0

    while generated < n_generate:
        batch_seqs = []
        while len(batch_seqs) < batch_size:
            seq = ''.join(random.choices(ALL_AAS, k=SEQ_LEN))
            if seq not in known:
                batch_seqs.append(seq)

        input_ids = torch.tensor(
            [encode_sequence(s) for s in batch_seqs],
            dtype=torch.long
        ).to(device)

        physchem = torch.stack(
            [encode_physchem(s) for s in batch_seqs]
        ).to(device)

        # Use midpoint time for generation
        time_norm = torch.full(
            (len(batch_seqs),), 0.5,
            dtype=torch.float32
        ).to(device)

        preds = model(input_ids, physchem, time_norm)

        pred_dg = (preds['delta_g'].cpu().numpy()
                   * HBD2Dataset.DG_STD + HBD2Dataset.DG_MEAN)
        pred_rmsd = (preds['rmsd'].cpu().numpy()
                     * HBD2Dataset.RMSD_STD + HBD2Dataset.RMSD_MEAN)

        for i, seq in enumerate(batch_seqs):
            all_results.append({
                'sequence':  seq,
                'pred_dg':   float(pred_dg[i]),
                'pred_rmsd': float(pred_rmsd[i]),
            })

        generated += batch_size
        if generated % 10000 == 0:
            print(f"  Generated {generated:,}/{n_generate:,}...")

    df = pd.DataFrame(all_results)

    # Filter by threshold
    candidates = df[df['pred_dg'] < dg_threshold].copy()
    print(f"\nPassing threshold ({dg_threshold}): {len(candidates):,} sequences")
    print(f"  ({len(candidates)/len(df)*100:.1f}% of generated)")

    # Rank by predicted ΔG
    candidates = candidates.sort_values('pred_dg').head(top_k)

    print(f"\nTop {len(candidates)} candidate sequences:")
    print(f"{'Rank':>4} | {'Sequence':>13} | {'Pred ΔG':>9} | {'Pred RMSD':>10}")
    print("-" * 50)
    for rank, (_, row) in enumerate(candidates.iterrows(), 1):
        print(f"{rank:>4} | {row['sequence']:>13} | "
              f"{row['pred_dg']:>9.2f} | {row['pred_rmsd']:>10.2f}")

    return candidates


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path = 'results/checkpoints/best_model.pt',
    model_config    = 'configs/model.yaml',
    config_path     = 'configs/training.yaml',
    excel_path      = 'data/raw/consolidated_cleaned.xlsx',
    generate        = True,
    n_generate      = 30000,
    top_k           = 30,
    save_results    = True,
):
    with open(config_path, 'r') as f:
        import yaml
        cfg = yaml.safe_load(f)

    device = cfg['training']['finetune']['device']
    seed   = cfg['training']['data']['random_seed']

    # Load model
    model = load_model(checkpoint_path, model_config, device)

    # Get test data
    _, _, test_loader = get_dataloaders(
        excel_path=excel_path,
        batch_size=256,
        random_seed=seed,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics, df = evaluate_test_set(model, test_loader, device)

    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"  Test ΔG RMSE      : {metrics['rmse']:.3f} kcal/mol")
    print(f"  Test MAE          : {metrics['mae']:.3f} kcal/mol")
    print(f"  Test R²           : {metrics['r2']:.3f}")
    print(f"  Physics Violations: {metrics['viol_rate']:.1f}%")
    print(f"  Test records      : {metrics['n_test']:,}")
    print("=" * 50)

    # Save predictions
    if save_results:
        Path('results').mkdir(exist_ok=True)
        df.to_csv('results/test_predictions.csv', index=False)
        print(f"\nPredictions saved to results/test_predictions.csv")

    # Generate visualizations
    print("\nGenerating figures...")
    plot_predictions(df, metrics)
    plot_training_history()

    # Generate new sequences
    if generate:
        candidates = generate_sequences(
            model, device,
            n_generate=n_generate,
            top_k=top_k,
            dg_threshold=-40.0,
            excel_path=excel_path,
        )
        if save_results:
            candidates.to_csv('results/candidate_sequences.csv', index=False)
            print(f"\nCandidates saved to results/candidate_sequences.csv")

    return metrics


if __name__ == "__main__":
    evaluate()