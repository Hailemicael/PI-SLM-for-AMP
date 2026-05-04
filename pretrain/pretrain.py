"""
pretrain/pretrain.py

Stage 1: Pretrain the PI-SLM on DBAASP sequences.

Goal: Teach the model what an AMP looks like before
      it sees any hBD-2 MD simulation data.

What this does:
    - Loads 32,244 sequences (AMPs + shuffled negatives)
    - Trains ONLY the binder classification head
    - Uses binary cross entropy loss
    - Saves pretrained weights for Stage 2 fine-tuning

Why pretraining matters:
    Our hBD-2 dataset has only 256 sequences.
    Pretraining on 32,244 sequences gives the model
    a strong foundation before fine-tuning on small data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.transformer import PISLM


# ─────────────────────────────────────────────────────────────
# AMINO ACID VOCABULARY
# Same as dataset.py — must be consistent
# ─────────────────────────────────────────────────────────────

AA_TO_IDX = {
    'A': 1,  'C': 2,  'D': 3,  'E': 4,
    'F': 5,  'G': 6,  'H': 7,  'I': 8,
    'K': 9,  'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16,
    'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    '<PAD>': 0, '<UNK>': 21
}

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')


def encode_and_pad(seq, max_len=50):
    """
    Encode sequence to integers and pad to max_len.
    DBAASP sequences vary in length (5-50 amino acids)
    so we pad shorter ones with zeros.

    Example:
        'RRWQ' padded to length 10 → [15,15,19,14,0,0,0,0,0,0]
    """
    encoded = [AA_TO_IDX.get(aa, 21) for aa in seq.upper()]
    # Truncate if longer than max_len
    encoded = encoded[:max_len]
    # Pad with zeros if shorter than max_len
    padded  = encoded + [0] * (max_len - len(encoded))
    return padded


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class DBAASPDataset(Dataset):
    def __init__(self, sequences, labels, max_len=50):
        self.sequences = sequences
        self.labels    = labels
        self.max_len   = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = self.sequences[idx]
        label = self.labels[idx]

        input_ids = torch.tensor(
            encode_and_pad(seq, self.max_len),
            dtype=torch.long
        )

        return {
            'input_ids': input_ids,
            'is_amp':    torch.tensor(label, dtype=torch.float32),
            'sequence':  seq,
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'is_amp':    torch.stack([b['is_amp']    for b in batch]),
        'sequences': [b['sequence'] for b in batch],
    }


# ─────────────────────────────────────────────────────────────
# LOAD DBAASP DATA
# ─────────────────────────────────────────────────────────────

def load_dbaasp(csv_path='data/raw/dbaasp/dbaasp_pretrain.csv'):
    """
    Load DBAASP sequences.
    Filters out sequences with invalid amino acids.
    """
    print(f"Loading DBAASP data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"  Raw sequences: {len(df)}")

    # Keep only valid sequences
    def is_valid(seq):
        if not isinstance(seq, str):
            return False
        seq = seq.strip().upper()
        return (
            5 <= len(seq) <= 50 and
            all(aa in VALID_AAS for aa in seq)
        )

    df = df[df['sequence'].apply(is_valid)].copy()
    df['sequence'] = df['sequence'].str.strip().str.upper()
    df = df.drop_duplicates(subset='sequence')

    print(f"  After filtering: {len(df)} sequences")
    print(f"  AMPs (is_amp=1): {(df['is_amp']==1).sum()}")
    print(f"  Non-AMPs (is_amp=0): {(df['is_amp']==0).sum()}")

    sequences = df['sequence'].tolist()
    labels    = df['is_amp'].astype(int).tolist()

    return sequences, labels


# ─────────────────────────────────────────────────────────────
# PRETRAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def pretrain(
    config_path    = 'configs/training.yaml',
    model_config   = 'configs/model.yaml',
    dbaasp_csv     = 'data/raw/dbaasp/dbaasp_pretrain.csv',
    save_path      = 'results/checkpoints/pretrained_model.pt',
):
    # ── Load config ───────────────────────────────────────────
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    pcfg    = cfg['training']['pretrain']
    device  = pcfg['device']
    epochs  = pcfg['epochs']
    lr      = pcfg['learning_rate']
    patience= pcfg['patience']
    bs      = pcfg['batch_size']

    print("=" * 70)
    print("STAGE 1 — PRETRAINING ON DBAASP")
    print("=" * 70)
    print(f"Device  : {device}")
    print(f"Epochs  : {epochs}")
    print(f"LR      : {lr}")
    print(f"Patience: {patience}")
    print(f"Batch   : {bs}")

    # ── Data ──────────────────────────────────────────────────
    sequences, labels = load_dbaasp(dbaasp_csv)

    # Split into train and validation
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels,
        test_size=0.15,
        random_state=42,
        stratify=labels,
    )

    print(f"\nData split:")
    print(f"  Train: {len(train_seqs)} sequences")
    print(f"  Val:   {len(val_seqs)} sequences")

    train_ds = DBAASPDataset(train_seqs, train_labels, max_len=50)
    val_ds   = DBAASPDataset(val_seqs,   val_labels,   max_len=50)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────
    # IMPORTANT: DBAASP sequences can be up to 50 amino acids long
    # But our model.yaml sets max_seq_len=13 for hBD-2
    # We temporarily override max_seq_len for pretraining
    with open(model_config, 'r') as f:
        mcfg = yaml.safe_load(f)
    mcfg['model']['max_seq_len'] = 50

    # Save temporary pretrain config
    pretrain_model_cfg = 'configs/model_pretrain.yaml'
    with open(pretrain_model_cfg, 'w') as f:
        yaml.dump(mcfg, f)

    model = PISLM(config_path=pretrain_model_cfg).to(device)

    # ── FREEZE everything except binder head ──────────────────
    # During pretraining we only train the binder classification head
    # and the token embedding — not the regression heads
    # This prevents the regression heads from learning garbage
    # on data that has no ΔG or RMSD labels
    for name, param in model.named_parameters():
        if 'heads.delta_g' in name:
            param.requires_grad = False
        if 'heads.rmsd' in name:
            param.requires_grad = False
        if 'heads.h_bonds' in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"(Regression heads frozen — only binder head trains)")

    # ── Loss ──────────────────────────────────────────────────
    # Class balance: roughly 50/50 AMPs and non-AMPs
    # pos_weight=1.0 means no correction needed
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-3,
    )

    # ── Save directory ────────────────────────────────────────
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────
    best_val_acc = 0.0
    patience_ctr = 0

    print("\n" + "-" * 50)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Val Acc':>8} | {'Val AUC':>8}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        epoch_losses = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels_b  = batch['is_amp'].to(device)

            predictions = model(input_ids)
            loss = bce(predictions['binder_logit'], labels_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # ── Validate ──────────────────────────────────────────
        model.eval()
        all_labels = []
        all_probs  = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                preds     = model(input_ids)
                probs     = torch.sigmoid(
                    preds['binder_logit']
                ).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(batch['is_amp'].numpy().tolist())

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        pred_labels = (all_probs >= 0.5).astype(int)
        val_acc = float((pred_labels == all_labels).mean())

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            val_auc = float('nan')

        mean_loss = np.mean(epoch_losses)
        print(f"{epoch:>6} | {mean_loss:>8.4f} | "
              f"{val_acc:>8.3f} | {val_auc:>8.3f}")

        # ── Save best ──────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc':     best_val_acc,
                'val_auc':          val_auc,
            }, save_path)
            print(f"         ✓ Saved  (acc={best_val_acc:.3f}  "
                  f"auc={val_auc:.3f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\n" + "=" * 70)
    print("PRETRAINING COMPLETE")
    print(f"  Best val accuracy : {best_val_acc:.3f}")
    print(f"  Saved to          : {save_path}")
    print("=" * 70)
    print("\nNext step: run main.py --mode finetune")


# ─────────────────────────────────────────────────────────────
# RUN STANDALONE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pretrain()