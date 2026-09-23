"""
Shared module for the Transformer-backbone experiment package: Transformer
backbone definitions + checkpoint-loading dispatch.

The architecture reuses the earlier HeteroscedasticTransformer /
MC_Transformer definitions verbatim (confirmed against the actual code:
the earlier manuscript draft's description of a "1D-Conv embedding" does
not match the implementation, which is actually an nn.Linear projection
plus a learnable additive positional encoding, with no Conv1d -- the code
is authoritative). Hyperparameters are likewise reused verbatim:
hidden_dim=64, nhead=4, dim_feedforward=128, num_layers=2, dropout=0.2,
log_sigma_min=-3.0, log_sigma_max=2.0, epochs=150, lr=1e-3, batch=256 --
these were determined under the old (leaky) protocol and were not
re-searched/re-tuned under the new protocol here; only the training-time
model-selection criterion changed to canonical_split's val_units (no
longer the test set).

MC_Transformer is a single-output, pure-MSE-trained version of
HeteroscedasticTransformer with the sigma head removed, used for
MC-Dropout (test-time dropout sampling) and the Fixed-variance/MSE rows --
its role exactly mirrors MC_LSTM on the LSTM side.
"""
import os
import torch
import torch.nn as nn

import common as C

SEQUENCE_LENGTH = C.SEQUENCE_LENGTH  # 30
T2_HIDDEN_DIM = 64
T2_NHEAD = 4
T2_DIM_FEEDFORWARD = 128
T2_NUM_LAYERS = 2
T2_DROPOUT = 0.2
T2_LOG_SIGMA_MIN = -3.0
T2_LOG_SIGMA_MAX = 2.0
T2_EPOCHS = 150
T2_LR = 0.001
T2_BATCH_SIZE = 256

BACKBONES = ['LSTM', 'Transformer']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
TRANSFORMER_DIR = os.path.join(RESULTS_DIR, 'leakfree_t2')
T2_CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'transformer')
os.makedirs(TRANSFORMER_DIR, exist_ok=True)
os.makedirs(T2_CKPT_DIR, exist_ok=True)


class HeteroscedasticTransformer(nn.Module):
    """Reuses the earlier training pipeline's definition verbatim."""
    def __init__(self, input_size, hidden_dim, dropout, seq_len, log_sigma_min, log_sigma_max):
        super().__init__()
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        self.input_proj = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=T2_NHEAD,
            dim_feedforward=T2_DIM_FEEDFORWARD, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=T2_NUM_LAYERS)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding
        x = self.transformer(x)
        feat = self.global_pool(x.permute(0, 2, 1)).squeeze(-1)
        feat = self.dropout(feat)
        mu = self.mu_head(feat)
        log_sigma = torch.clamp(self.log_sigma_head(feat), self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma


class MC_Transformer(nn.Module):
    """Single-output MSE version of HeteroscedasticTransformer with the
    sigma head removed, mirroring MC_LSTM (mc_dropout_model.py): used for
    MC-Dropout test-time sampling and the Fixed-variance/MSE rows."""
    def __init__(self, input_size, hidden_dim, dropout, seq_len):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        self.input_proj = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=T2_NHEAD,
            dim_feedforward=T2_DIM_FEEDFORWARD, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=T2_NUM_LAYERS)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.mu = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding
        x = self.transformer(x)
        feat = self.global_pool(x.permute(0, 2, 1)).squeeze(-1)
        feat = self.dropout(feat)
        return self.mu(feat)


def load_checkpoint_model_t2(backbone, ckpt_path, device):
    """Backbone-dispatching checkpoint loader for NLL (Heteroscedastic*)
    models. The LSTM branch reuses common.load_checkpoint_model directly;
    the Transformer branch additionally needs seq_len (fixed =
    SEQUENCE_LENGTH, shared window length across both backbones)."""
    if backbone == 'LSTM':
        return C.load_checkpoint_model(ckpt_path, device)
    elif backbone == 'Transformer':
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = HeteroscedasticTransformer(
            ckpt['input_dim'], ckpt['hidden_dim'], ckpt['dropout'], SEQUENCE_LENGTH,
            ckpt['log_sigma_min'], ckpt['log_sigma_max']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model
    raise ValueError(backbone)


def load_checkpoint_mc_model_t2(backbone, ckpt_path, device):
    """Backbone-dispatching checkpoint loader for MC-Dropout/MSE (MC_*) models."""
    if backbone == 'LSTM':
        import mc_dropout_model as S1
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = S1.MC_LSTM(ckpt['input_dim'], ckpt['hidden_dim'], ckpt['dropout']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model
    elif backbone == 'Transformer':
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = MC_Transformer(ckpt['input_dim'], ckpt['hidden_dim'], ckpt['dropout'], SEQUENCE_LENGTH).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model
    raise ValueError(backbone)


def nll_ckpt_path(backbone, ds, seed, extra=False):
    tag = 'extraseed' if extra else 'seed'
    if backbone == 'LSTM':
        # reuses the existing leakage-free LSTM checkpoint (no retraining)
        d = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
        return os.path.join(d, f"{ds}_LSTM_{tag}{seed}.pt")
    return os.path.join(T2_CKPT_DIR, f"{ds}_Transformer_{tag}{seed}.pt")


def mc_ckpt_path(backbone, ds, seed):
    if backbone == 'LSTM':
        d = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
        return os.path.join(d, f"{ds}_MCDropoutMSE_seed{seed}.pt")
    return os.path.join(T2_CKPT_DIR, f"{ds}_MCDropoutMSE_Transformer_seed{seed}.pt")
