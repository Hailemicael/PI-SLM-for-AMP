"""
model/transformer.py

Custom Small Language Model for AMP binding affinity prediction.
Built entirely from scratch - no pretrained model dependency.

Architecture:
    Token Embedding + Physchem Features (5) + Time Signal
    → Positional Encoding → 4x Transformer Layers
    → Global Average Pooling → 4 Output Heads
"""

import torch
import torch.nn as nn
import math
import yaml

# ─────────────────────────────────────────────────────────────
# PHYSICOCHEMICAL PROPERTIES — 5 features per amino acid
# hydrophobicity, charge, molecular_weight, polarity, volume
# ─────────────────────────────────────────────────────────────

PHYSCHEM_DIM = 5

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
# COMPONENT 1 — TOKEN EMBEDDING
# ─────────────────────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_dim)


# ─────────────────────────────────────────────────────────────
# COMPONENT 2 — POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────
# COMPONENT 3 — TRANSFORMER ENCODER LAYER
# ─────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed      = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + ff_out

        return x


# ─────────────────────────────────────────────────────────────
# COMPONENT 4 — OUTPUT HEADS
# ─────────────────────────────────────────────────────────────

class OutputHeads(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.delta_g = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.rmsd = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        self.h_bonds = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return {
            'delta_g':      self.delta_g(x).squeeze(-1),
            'rmsd':         self.rmsd(x).squeeze(-1),
            'h_bonds':      self.h_bonds(x).squeeze(-1),
        }


# ─────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────

class PISLM(nn.Module):
    """
    Physics-Informed Small Language Model for AMP design.

    Inputs:
        input_ids : (batch, 13)    — amino acid integers
        physchem  : (batch, 13, 5) — 5 physicochemical features
        time_norm : (batch,)       — normalized simulation time [0,1]

    Output:
        dict with delta_g, rmsd, h_bonds, binder_logit
        each of shape (batch,)
    """

    def __init__(self, config_path='configs/model.yaml'):
        super().__init__()

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)['model']

        self.embed_dim   = cfg['embed_dim']
        self.num_layers  = cfg['num_layers']
        self.num_heads   = cfg['num_heads']
        self.ff_dim      = cfg['ff_dim']
        self.dropout     = cfg['dropout']
        self.vocab_size  = cfg['vocab_size']
        self.max_seq_len = cfg['max_seq_len']

        # Component 1 — Token Embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim
        )

        # Physicochemical projection
        # Combines token embedding (128) with physchem (5) → 128
        self.physchem_proj = nn.Linear(
            self.embed_dim + PHYSCHEM_DIM,
            self.embed_dim
        )

        # Time projection
        self.time_proj = nn.Linear(1, self.embed_dim)

        # Component 2 — Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_dim=self.embed_dim,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout
        )

        # Component 3 — Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(self.embed_dim)

        # Component 4 — Output Heads
        self.heads = OutputHeads(embed_dim=self.embed_dim)

        # Initialize weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"PI-SLM initialized: {total_params:,} parameters")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids, physchem=None, time_norm=None):
        # Step 1: Token embedding (batch, 13) → (batch, 13, 128)
        x = self.token_embedding(input_ids)

        # Step 2: Add physicochemical features
        if physchem is not None:
            x = torch.cat([x, physchem], dim=-1)  # (batch, 13, 133)
            x = self.physchem_proj(x)              # (batch, 13, 128)

        # Step 3: Add time signal
        if time_norm is not None:
            t = time_norm.unsqueeze(-1).unsqueeze(-1)
            t = self.time_proj(t)
            t = t.expand(-1, x.size(1), -1)
            x = x + t

        # Step 4: Positional encoding
        x = self.pos_encoding(x)

        # Step 5: Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Step 6: Global average pooling
        x = self.final_norm(x)
        x = x.mean(dim=1)

        # Step 7: Output heads
        return self.heads(x)


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing model...")

    model = PISLM(config_path='configs/model.yaml')
