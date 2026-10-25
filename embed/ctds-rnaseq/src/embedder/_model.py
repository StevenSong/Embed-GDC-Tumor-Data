"""ctds-rnaseq: a BulkRNABert-architecture bulk RNA-seq encoder, float32 TPM in, (B, 256) out.

Copied from uc-cdis/omics-encoder `kserve-demo/` (Apache-2.0, Copyright 2024
University of Chicago; see NOTICE): `model.py`, the tokenizer and pooling of
`wrapped.py`, and `data.load_binner`. Those are the frozen copies of the code the
checkpoint was trained with, not omics-encoder's current `src/models/bert.py`. Two
consequences that look like bugs and are not:

* `embedding_norm` is always applied, though the checkpoint's `run_config.json` says
  `no_embedding_norm: true` -- the training code ignored that flag.
* The feed-forward is `fc2(fc1(h))` with no activation between, as trained.

Trimmed to inference: no gene2vec init, weight init, MLM loss or `forward`. The
module tree, and so the state_dict keys, is unchanged, and `lm_head` is kept only
so the checkpoint loads strictly.

The embedding is fixed to what the served model and the collaborator's offline
pipeline produce: the last encoder block's output, before `final_norm`, mean-pooled
over all genes. Absent genes go in as MASK_ID and are averaged in with the rest.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BINS = 64
PAD_ID = N_BINS
MASK_ID = N_BINS + 1
VOCAB_SIZE = N_BINS + 2

# the files a weights directory holds; the Dockerfile bakes them in at WEIGHTS_DIR
WEIGHTS_DIR = "/opt/ctds-rnaseq"
WEIGHTS = "model.pt"
BINNER = "binner.npz"
RUN_CONFIG = "run_config.json"
GENES = "genes.tsv"

LAYER = "last"  # of the 4 encoder blocks, before final_norm
POOLING = "mean"  # over all genes, absent ones included


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden dim {hidden_dim} must be divisible by num heads {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        h = self.norm1(x)
        qkv = self.qkv(h).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        h = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        h = h.transpose(1, 2).reshape(B, L, D)
        x = x + self.out_proj(h)

        h = self.norm2(x)
        h = self.fc2(self.fc1(h))
        return x + h


class BulkRNABert(nn.Module):
    def __init__(
        self,
        num_genes: int,
        hidden_dim: int = 256,
        embed_dim: int = 200,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 512,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.token_embedding = nn.Embedding(VOCAB_SIZE, hidden_dim)
        self.position_embedding = nn.Embedding(num_genes, embed_dim)
        self.gene_embed_adapt_layer = nn.Linear(embed_dim, hidden_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            EncoderBlock(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)  # unused: we pool before it
        self.lm_head = nn.Linear(hidden_dim, N_BINS)  # unused: MLM head
        self.register_buffer("gene_ids", torch.arange(num_genes), persistent=False)


class Encoder(nn.Module):
    """BulkRNABert plus its tokenizer: (B, G) float32 TPM, NaN = absent -> (B, D)."""

    def __init__(self, model: BulkRNABert, binner: dict):
        super().__init__()
        self.model = model.eval()
        self.n_bins = int(binner["n_bins"])
        self.kind = str(binner["kind"])
        if self.kind == "global_linear":
            self.norm_factor = float(binner["norm_factor"])
            # np.linspace is float64 and np.digitize compares in float64; keep both.
            e = np.linspace(0.0, 1.0, self.n_bins)
        elif self.kind == "per_gene_quantile":
            e = np.ascontiguousarray(binner["edges"])  # (G, n_bins - 2), as saved
            if e.shape[0] != model.num_genes:
                raise ValueError(
                    f"binner has {e.shape[0]} genes, model {model.num_genes}"
                )
        else:
            raise ValueError(f"unrecognised binner {self.kind!r}")
        # Compare in the saved edges' dtype, exactly as numpy promotes.
        self.register_buffer("edges", torch.from_numpy(e))

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        absent = torch.isnan(x)
        filled = torch.where(absent, torch.zeros_like(x), x)
        if self.kind == "global_linear":
            # float32 math, as in numpy (python-float divisor does not upcast)
            xl = torch.log10(1.0 + filled) / self.norm_factor
            # right=True == np.digitize default: edges[i-1] <= x < edges[i]
            tok = torch.bucketize(xl.to(self.edges.dtype), self.edges, right=True)
            tok = tok.clamp(0, self.n_bins - 1)
        else:
            xl = torch.log10(1.0 + filled).to(self.edges.dtype)
            # per-row search: sequence (G, E), values (G, B); side="right" semantics
            tok = (
                torch.searchsorted(
                    self.edges, xl.transpose(0, 1).contiguous(), right=True
                ).transpose(0, 1)
                + 1
            )
        tok = tok.masked_fill(filled == 0.0, 0)
        return tok.masked_fill(absent, MASK_ID)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BulkRNABert's embedding stage, then the encoder blocks; never final_norm
        # or lm_head, whose (B, G, 64) logits would be computed and thrown away.
        m = self.model
        x = m.token_embedding(self.tokenize(x))
        g = m.gene_embed_adapt_layer(m.position_embedding(m.gene_ids))
        x = m.embedding_norm(x + g.unsqueeze(0))
        for blk in m.layers:
            x = blk(x)
        return x.mean(dim=1).float()


def load_encoder(weights_dir: Path, num_genes: int) -> Encoder:
    cfg = json.loads((weights_dir / RUN_CONFIG).read_text())
    model = BulkRNABert(
        num_genes=num_genes,
        hidden_dim=cfg["hidden_dim"],
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
    )
    state = torch.load(weights_dir / WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict: a vocabulary of the wrong size fails here
    with np.load(weights_dir / BINNER, allow_pickle=False) as d:
        binner = {k: d[k] for k in d.files}
    return Encoder(model, binner).eval()
