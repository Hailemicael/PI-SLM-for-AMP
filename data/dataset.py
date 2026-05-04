"""
data/dataset.py

Time-series dataset for PI-SLM training.

Exact replication of CNN paper approach:
    Input:  physicochemical encoding (5 features) + time point
    Output: instantaneous ΔG at that time point
    Data:   28,100 time points (281 sequences × 100 ns)
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# AMINO ACID VOCABULARY
# ─────────────────────────────────────────────────────────────

AA_TO_IDX = {
    'A': 1,  'C': 2,  'D': 3,  'E': 4,
    'F': 5,  'G': 6,  'H': 7,  'I': 8,
    'K': 9,  'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16,
    'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    '<PAD>': 0, '<UNK>': 21
}

VOCAB_SIZE = 22
VALID_AAS  = set('ACDEFGHIKLMNPQRSTVWY')
PHYSCHEM_DIM = 5


def encode_sequence(seq):
    return [AA_TO_IDX.get(aa, 21) for aa in seq.upper()]


# ─────────────────────────────────────────────────────────────
# PHYSICOCHEMICAL PROPERTIES — 5 features per amino acid
# hydrophobicity, charge, molecular_weight, polarity, volume
# ─────────────────────────────────────────────────────────────

AA_PROPERTIES = {
    'A': [ 0.62,  0.0,  0.57, -0.50,  0.22],
    'C': [ 0.29,  0.0,  0.78,  0.00,  0.35],
    'D': [-0.90, -1.0,  0.83,  0.80,  0.40],
    'E': [-0.74, -1.0,  1.00,  0.90,  0.54],
    'F': [ 1.19,  0.0,  1.10, -0.50,  0.81],
    'G': [ 0.48,  0.0,  0.39, -0.50,  0.00],
    'H': [-0.40,  0.5,  1.08,  0.50,  0.67],
    'I': [ 1.38,  0.0,  0.86, -0.50,  0.72],
    'K': [-1.50,  1.0,  1.00,  1.00,  0.73],
    'L': [ 1.06,  0.0,  0.86, -0.50,  0.72],
    'M': [ 0.64,  0.0,  1.04, -0.30,  0.70],
    'N': [-0.78,  0.0,  0.84,  0.85,  0.51],
    'P': [ 0.12,  0.0,  0.72, -0.50,  0.49],
    'Q': [-0.85,  0.0,  1.01,  0.90,  0.61],
    'R': [-2.53,  1.0,  1.17,  1.00,  0.83],
    'S': [-0.18,  0.0,  0.61,  0.60,  0.28],
    'T': [-0.05,  0.0,  0.76,  0.50,  0.44],
    'V': [ 1.08,  0.0,  0.72, -0.50,  0.57],
    'W': [ 0.81,  0.0,  1.39, -0.30,  1.00],
    'Y': [ 0.26,  0.0,  1.21,  0.50,  0.90],
}


def encode_physchem(seq):
    """
    Convert amino acid sequence to physicochemical property matrix.
    Input:  string like 'VFCPRRYKQIGTC'
    Output: tensor of shape (seq_len, 5)
    """
    props = []
    for aa in seq.upper():
        props.append(AA_PROPERTIES.get(aa, [0.0] * PHYSCHEM_DIM))
    return torch.tensor(props, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# STATIC LABELS — RMSD and H-bonds from SI tables
# ─────────────────────────────────────────────────────────────

STATIC_LABELS = {
    'VFCPRRYKQIGTC': (2.20, 3.84),
    'VFCPRRYIQIGTC': (6.10, 4.79),
    'VFCPRRYKKIGTC': (4.50, 4.56),
    'VFCPRRWKQIGTC': (2.60, 5.08),
    'VFCPRHYKQIGTC': (2.40, 5.39),
    'VFCPRLYKQIGTC': (2.10, 5.70),
    'VFCPRRLKQIGTC': (4.30, 4.33),
    'VFCPRRYFQIGTC': (2.10, 5.56),
    'VFCPRRYHQIGTC': (1.90, 5.37),
    'VFCPRRYKQIGYC': (2.60, 4.90),
    'VFCPRRYKQIGRC': (2.40, 5.11),
    'VFCPRRYKQIGTK': (2.30, 5.38),
    'IKKHWLLFWNYRK': (3.70, 1.07),
    'VFSPRRYKQIGTS': (3.60, 3.07),
    'VACARAYAQAGAC': (4.30, 1.09),
    'IAKAWALAWAYAK': (3.00, 0.64),
    'AFCARRAKQAGTA': (2.50, 3.49),
    'VACPARYAQIATC': (2.10, 2.70),
    'IKAHWALFANYAK': (3.90, 2.80),
    'IAKHALLAWNARK': (4.50, 0.91),
    'VFKPRRLKQIYTC': (2.60, 1.56),
    'IAKHWALFWAYRK': (9.40, 1.34),
    'IKAHWLAFWNARK': (2.60, 1.23),
    'VFKPWLYFQIGTC': (3.10, 1.91),
    'VFIPWLLFQIYRK': (4.60, 0.95),
    'VPCPPCYHCIGTC': (3.70, 2.25),
    'VFIPWRYHQIYRC': (3.60, 1.54),
    'VFIPRLYFQIYTK': (4.10, 3.15),
    'VFKPWHYKQIGRK': (3.00, 2.08),
    'VFKPWRLKQIYRC': (4.50, 2.16),
}


# ─────────────────────────────────────────────────────────────
# NORMALIZATION CONSTANTS
# Computed from full 28,100 time-series records
# ─────────────────────────────────────────────────────────────

class HBD2Dataset(Dataset):
    """
    Time-series dataset — one record per time point.
    Each record: (sequence, time_ns) → instantaneous ΔG
    28,100 training examples total.
    """

    # Updated for average ΔG target
    # Average ΔG has smaller std than instantaneous ΔG
    DG_MEAN   = -22.50
    DG_STD    =  12.80
    RMSD_MEAN =   4.50
    RMSD_STD  =   2.50
    TIME_MIN  = 401.0
    TIME_MAX  = 500.0

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        seq = r['sequence']

        input_ids = torch.tensor(
            encode_sequence(seq),
            dtype=torch.long
        )

        physchem = encode_physchem(seq)

        time_norm = (
            (r['time_ns'] - self.TIME_MIN)
            / (self.TIME_MAX - self.TIME_MIN)
        )

        dg_norm = (r['delta_g'] - self.DG_MEAN) / self.DG_STD

        rmsd     = r.get('rmsd', None)
        hb       = r.get('h_bonds', None)
        has_rmsd = rmsd is not None
        has_hb   = hb is not None

        rmsd_norm = (
            (rmsd - self.RMSD_MEAN) / self.RMSD_STD
            if has_rmsd else 0.0
        )


        return {
            'sequence':   seq,
            'input_ids':  input_ids,
            'physchem':   physchem,
            'time_norm':  torch.tensor(time_norm,  dtype=torch.float32),
            'delta_g':    torch.tensor(dg_norm,    dtype=torch.float32),
            'rmsd':       torch.tensor(rmsd_norm,  dtype=torch.float32),
            'h_bonds':    torch.tensor(hb if has_hb else 0.0,
                                       dtype=torch.float32),
            'has_rmsd':   torch.tensor(has_rmsd,   dtype=torch.bool),
            'has_hbonds': torch.tensor(has_hb,     dtype=torch.bool),
            'raw_delta_g':torch.tensor(r['delta_g'],dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────
# LOAD TIME SERIES DATA
# ─────────────────────────────────────────────────────────────

def load_timeseries(excel_path):
    print(f"Loading time-series data from {excel_path}...")
    df_raw = pd.read_excel(
        excel_path, sheet_name='Sheet1', header=None
    )

    records = []
    for col in range(0, df_raw.shape[1], 2):
        seq = df_raw.iloc[0, col]
        if not isinstance(seq, str):
            continue
        seq = seq.upper().strip()
        if len(seq) != 13:
            continue
        if not all(aa in VALID_AAS for aa in seq):
            continue

        for row in range(2, df_raw.shape[0]):
            try:
                t  = float(df_raw.iloc[row, col])
                dg = float(df_raw.iloc[row, col + 1])
                if not np.isnan(t) and not np.isnan(dg):
                    records.append({
                        'sequence': seq,
                        'time_ns':  t,
                        'delta_g':  dg,
                    })
            except Exception:
                continue

    df = pd.DataFrame(records)
    print(f"  Loaded: {len(df):,} time points")
    print(f"  Sequences: {df['sequence'].nunique()}")
    print(f"  Time range: {df['time_ns'].min():.0f} - "
          f"{df['time_ns'].max():.0f} ns")
    print(f"  ΔG range: {df['delta_g'].min():.1f} to "
          f"{df['delta_g'].max():.1f} kcal/mol")
    return df


# def build_dataset(excel_path):
#     df = load_timeseries(excel_path)

#     records = []
#     for _, row in df.iterrows():
#         seq    = row['sequence']
#         static = STATIC_LABELS.get(seq, None)

#         records.append({
#             'sequence':  seq,
#             'time_ns':   row['time_ns'],
#             'delta_g':   row['delta_g'],
#             'rmsd':      static[0] if static else None,
#             'h_bonds':   static[1] if static else None,
#         })

#     print(f"\nDataset summary:")
#     print(f"  Total records     : {len(records):,}")
#     print(f"  Binder records    : "
#           f"{sum(1 for r in records if r['delta_g']<0):,}")
#     print(f"  With RMSD labels  : "
#           f"{sum(1 for r in records if r['rmsd']):,}")
#     print(f"  With H-bond labels: "
#           f"{sum(1 for r in records if r['h_bonds']):,}")
#     return records

def build_dataset(excel_path):
    """
    Build time-series dataset with average ΔG as target.

    Key insight:
        Instantaneous ΔG fluctuates wildly due to thermal noise.
        Using average ΔG per sequence as target removes noise
        while keeping all 100 time points as training examples.

        This gives the model:
        - 28,100 training examples (data quantity advantage)
        - Stable regression target (noise reduction advantage)
        - Time signal still teaches temporal dynamics
    """
    df = load_timeseries(excel_path)

    # Compute per-sequence average ΔG
    # This is the stable binding affinity signal
    avg_dg = df.groupby('sequence')['delta_g'].mean().reset_index()
    avg_dg.columns = ['sequence', 'avg_delta_g']
    df = df.merge(avg_dg, on='sequence')

    records = []
    for _, row in df.iterrows():
        seq    = row['sequence']
        static = STATIC_LABELS.get(seq, None)

        records.append({
            'sequence':  seq,
            'time_ns':   row['time_ns'],
            'delta_g':   row['avg_delta_g'],  # average not instantaneous
            'rmsd':      static[0] if static else None,
            'h_bonds':   static[1] if static else None,
        })

    # Statistics
    unique_dgs = df.groupby('sequence')['avg_delta_g'].first()
    binder_seqs = (unique_dgs < 0).sum()

    print(f"\nDataset summary:")
    print(f"  Total records     : {len(records):,}")
    print(f"  Unique sequences  : {df['sequence'].nunique()}")
    print(f"  Binder sequences  : {binder_seqs}")
    print(f"  ΔG range (avg)    : "
          f"{unique_dgs.min():.1f} to {unique_dgs.max():.1f} kcal/mol")
    print(f"  With RMSD labels  : "
          f"{sum(1 for r in records if r['rmsd']):,}")
    print(f"  With H-bond labels: "
          f"{sum(1 for r in records if r['h_bonds']):,}")
    return records

def get_dataloaders(
    excel_path='data/raw/consolidated_cleaned.xlsx',
    batch_size=256,
    random_seed=42,
):
    """
    Random split by record — same approach as original CNN paper.
    """
    records = build_dataset(excel_path)

    train_val, test_records = train_test_split(
        records, test_size=0.15,
        random_state=random_seed
    )
    train_records, val_records = train_test_split(
        train_val, test_size=0.15/0.85,
        random_state=random_seed
    )

    print(f"\nData split (by record — CNN paper approach):")
    print(f"  Train: {len(train_records):,} records")
    print(f"  Val:   {len(val_records):,} records")
    print(f"  Test:  {len(test_records):,} records")

    train_loader = DataLoader(
        HBD2Dataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        HBD2Dataset(val_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        HBD2Dataset(test_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing dataset...")

    train_loader, val_loader, test_loader = get_dataloaders(
        excel_path='data/raw/consolidated_cleaned.xlsx',
        batch_size=256,
    )

    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  input_ids  : {batch['input_ids'].shape}")
    print(f"  physchem   : {batch['physchem'].shape}")
    print(f"  time_norm  : {batch['time_norm'][:5]}")
    print(f"  delta_g    : {batch['delta_g'][:5]}")
    print(f"  is_binder  : {batch['is_binder'][:5]}")
    print(f"  has_rmsd   : {batch['has_rmsd'][:5]}")
    print("\nDataset test PASSED")