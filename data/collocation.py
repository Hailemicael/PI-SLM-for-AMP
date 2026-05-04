import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data.dataset import (
    AA_TO_IDX,
    encode_sequence,
    AA_PROPERTIES,
    encode_physchem,
    build_dataset,
    HBD2Dataset,
)

# All 20 amino acids
ALL_AAS = list('ACDEFGHIKLMNPQRSTVWY')
SEQ_LEN = 13


# ─────────────────────────────────────────────────────────────
# PHYSICOCHEMICAL CALCULATIONS FROM SEQUENCE ALONE
# These do not require MD simulation
# ─────────────────────────────────────────────────────────────

CATIONIC  = set('KRH')
ANIONIC   = set('DE')
HYDROPHOBIC = set('VILFWMA')


def net_charge(seq):
    """
    Compute net charge of sequence.
    Positive charge comes from K, R, H.
    Negative charge comes from D, E.
    hBD-2 wildtype has net charge +4.
    """
    pos = sum(1 for aa in seq if aa in CATIONIC)
    neg = sum(1 for aa in seq if aa in ANIONIC)
    return pos - neg


def hydrophobic_fraction(seq):
    """
    Fraction of hydrophobic residues.
    AMPs typically have 30-60% hydrophobic residues.
    """
    return sum(1 for aa in seq if aa in HYDROPHOBIC) / len(seq)


def sequence_similarity(seq1, seq2):
    """
    Simple position-wise similarity between two sequences.
    Returns fraction of matching positions.
    """
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1)


def estimate_binder(seq, known_strong_binders, known_non_binders):
    """
    Estimate whether a sequence is likely a binder based on
    physicochemical properties and similarity to known sequences.

    This is a rough heuristic — NOT a replacement for MD simulation.
    Used ONLY to set binder_mask for physics constraint application.

    Rules based on domain knowledge from your paper:
    1. Net charge should be positive (hBD-2 is cationic)
    2. Hydrophobicity should be 30-60%
    3. Similarity to known strong binders suggests binding
    4. Similarity to known non-binders suggests non-binding
    """
    charge = net_charge(seq)
    hydro  = hydrophobic_fraction(seq)

    # Rule 1: Strong negative charge — unlikely binder
    if charge < 0:
        return 0

    # Rule 2: Extreme hydrophobicity — unlikely binder
    if hydro > 0.8 or hydro < 0.1:
        return 0

    # Rule 3: Similar to known strong binders — likely binder
    max_sim_binder = max(
        sequence_similarity(seq, s) for s in known_strong_binders
    ) if known_strong_binders else 0

    # Rule 4: Similar to known non-binders — likely non-binder
    max_sim_nonbinder = max(
        sequence_similarity(seq, s) for s in known_non_binders
    ) if known_non_binders else 0

    if max_sim_binder > max_sim_nonbinder and max_sim_binder > 0.4:
        return 1

    # Rule 5: Good charge and hydrophobicity profile
    if 2 <= charge <= 8 and 0.3 <= hydro <= 0.6:
        return 1

    return 0


def estimate_pseudo_dg(seq, known_strong_binders):
    """
    Estimate a rough pseudo ΔG for collocation sequences.

    NOT used as a regression target.
    Used only to make the physics constraints meaningful.

    More negative = stronger predicted binding.
    Based purely on physicochemical properties.
    """
    charge = net_charge(seq)
    hydro  = hydrophobic_fraction(seq)

    # Similarity to wildtype VFCPRRYKQIGTC (ΔG = -40.17)
    wildtype = 'VFCPRRYKQIGTC'
    sim_to_wt = sequence_similarity(seq, wildtype)

    # Similarity to strongest binder K25F (ΔG = -58.6)
    k25f = 'VFCPRRYFQIGTC'
    sim_to_best = sequence_similarity(seq, k25f)

    if known_strong_binders:
        max_sim = max(
            sequence_similarity(seq, s) for s in known_strong_binders
        )
    else:
        max_sim = 0

    # Rough estimate
    # Base: -20 kcal/mol
    # Charge contribution: cationic charge improves binding
    # Hydrophobicity: moderate hydrophobicity helps
    # Similarity: closer to known binders → stronger binding
    base    = -20.0
    charge_contrib = min(charge * 2.0, 10.0)
    hydro_contrib  = -10.0 * (1 - abs(hydro - 0.45) / 0.45)
    sim_contrib    = -20.0 * max_sim

    pseudo_dg = base - charge_contrib + hydro_contrib + sim_contrib

    # Clamp to physical range
    pseudo_dg = max(-58.6, min(0.0, pseudo_dg))

    return pseudo_dg


# ─────────────────────────────────────────────────────────────
# GENERATE COLLOCATION SEQUENCES
# ─────────────────────────────────────────────────────────────

def generate_collocation_sequences(n=10000, random_seed=42):
    """
    Generate n synthetic sequences for collocation training.

    Strategy:
        50% purely random sequences
        25% sequences close to known strong binders (1-3 mutations)
        25% sequences close to wildtype (1-2 mutations)

    This ensures coverage of:
        - Novel sequence space (random)
        - Near-binder space (mutations of strong binders)
        - Near-wildtype space (mutations of wildtype)
    """
    random.seed(random_seed)

    # Get known sequences from training data
    all_data = build_dataset()
    known_seqs    = set(d['sequence'] for d in all_data)
    strong_binders = [
        d['sequence'] for d in all_data
        if d['delta_g'] <= -40.0
    ]
    non_binders = [
        d['sequence'] for d in all_data
        if d['delta_g'] == 0.0
    ]

    print(f"Generating {n:,} collocation sequences...")
    print(f"  Reference strong binders : {len(strong_binders)}")
    print(f"  Reference non-binders    : {len(non_binders)}")

    sequences = []
    seen      = set(known_seqs)

    n_random   = n // 2
    n_mutated  = n // 4
    n_wt_mutated = n - n_random - n_mutated

    # 1. Purely random sequences
    attempts = 0
    while len(sequences) < n_random and attempts < n_random * 10:
        seq = ''.join(random.choices(ALL_AAS, k=SEQ_LEN))
        if seq not in seen:
            sequences.append(seq)
            seen.add(seq)
        attempts += 1

    # 2. Mutations of strong binders
    attempts = 0
    while len(sequences) < n_random + n_mutated and attempts < n_mutated * 10:
        if not strong_binders:
            break
        parent = random.choice(strong_binders)
        seq    = list(parent)
        n_muts = random.randint(1, 3)
        positions = random.sample(range(SEQ_LEN), n_muts)
        for pos in positions:
            seq[pos] = random.choice(ALL_AAS)
        seq = ''.join(seq)
        if seq not in seen:
            sequences.append(seq)
            seen.add(seq)
        attempts += 1

    # 3. Mutations of wildtype
    wildtype = 'VFCPRRYKQIGTC'
    attempts = 0
    while len(sequences) < n and attempts < n_wt_mutated * 10:
        seq    = list(wildtype)
        n_muts = random.randint(1, 2)
        positions = random.sample(range(SEQ_LEN), n_muts)
        for pos in positions:
            seq[pos] = random.choice(ALL_AAS)
        seq = ''.join(seq)
        if seq not in seen:
            sequences.append(seq)
            seen.add(seq)
        attempts += 1

    print(f"  Generated: {len(sequences):,} unique sequences")

    # Assign pseudo-labels
    records = []
    for seq in sequences:
        is_binder  = estimate_binder(seq, strong_binders, non_binders)
        pseudo_dg  = estimate_pseudo_dg(seq, strong_binders)
        pseudo_rmsd = abs(pseudo_dg) / 10.0  # rough estimate

        records.append({
            'sequence':   seq,
            'is_binder':  is_binder,
            'pseudo_dg':  pseudo_dg,
            'pseudo_rmsd':pseudo_rmsd,
        })

    n_binders = sum(r['is_binder'] for r in records)
    print(f"  Estimated binders    : {n_binders:,}")
    print(f"  Estimated non-binders: {len(records)-n_binders:,}")

    return records


# ─────────────────────────────────────────────────────────────
# PYTORCH DATASET FOR COLLOCATION POINTS
# ─────────────────────────────────────────────────────────────

class CollocationDataset(Dataset):
    """
    Dataset of synthetic collocation sequences.

    Key difference from HBD2Dataset:
        - No real ΔG or RMSD labels
        - Only pseudo-labels for binder_mask
        - Training uses PHYSICS LOSS ONLY on these sequences
        - Data loss is never computed on collocation points
    """

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        seq = r['sequence']

        input_ids = torch.tensor(
            encode_sequence(seq),
            dtype=torch.long
        )
        physchem = encode_physchem(seq)

        # Normalize pseudo-labels same way as real data
        pseudo_dg_norm = (
            (r['pseudo_dg'] - HBD2Dataset.DG_MEAN)
            / HBD2Dataset.DG_STD
        )
        pseudo_rmsd_norm = (
            (r['pseudo_rmsd'] - HBD2Dataset.RMSD_MEAN)
            / HBD2Dataset.RMSD_STD
        )

        return {
            'sequence':    seq,
            'input_ids':   input_ids,
            'physchem':    physchem,
            'is_binder':   torch.tensor(
                               r['is_binder'],
                               dtype=torch.float32
                           ),
            'pseudo_dg':   torch.tensor(
                               pseudo_dg_norm,
                               dtype=torch.float32
                           ),
            'pseudo_rmsd': torch.tensor(
                               pseudo_rmsd_norm,
                               dtype=torch.float32
                           ),
        }


def get_collocation_loader(n=10000, batch_size=32, random_seed=42):
    """
    Build collocation dataloader.
    """
    records = generate_collocation_sequences(n=n, random_seed=random_seed)
    dataset = CollocationDataset(records)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"  Collocation loader: {len(dataset):,} sequences "
          f"in {len(loader):,} batches")
    return loader


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing collocation dataset...")

    loader = get_collocation_loader(n=1000, batch_size=16)
    batch  = next(iter(loader))

    print(f"\nSample batch:")
    print(f"  input_ids shape : {batch['input_ids'].shape}")
    print(f"  physchem shape  : {batch['physchem'].shape}")
    print(f"  is_binder       : {batch['is_binder'][:5]}")
    print(f"  pseudo_dg       : {batch['pseudo_dg'][:5]}")

    print("\nCollocation dataset test PASSED")