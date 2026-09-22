"""
T2 升级实验包共用模块：Transformer 骨干定义 + checkpoint 加载调度。

架构逐字复用 E3_run_save_ece.py 的 HeteroscedasticTransformer / MC_Transformer
定义（2026-09-18 用户确认：the earlier manuscript draft 描述的"1D-Conv embedding"与实际代码
不符，实际是 nn.Linear 投影 + 可学习加性位置编码，按代码为准，不引入 Conv1d）。
超参同样逐字复用：hidden_dim=64, nhead=4, dim_feedforward=128, num_layers=2,
dropout=0.2, log_sigma_min=-3.0, log_sigma_max=2.0, epochs=150, lr=1e-3,
batch=256 —— 在旧（泄漏）协议下确定，本次未在新协议下重新搜索/调参，
只是训练时的模型选择判据换成 canonical_split 的 val_units（不用测试集）。

MC_Transformer 是 HeteroscedasticTransformer 去掉 sigma 头、纯 MSE 训练的单输出
版本，用于 MC-Dropout（测试时 dropout 采样）和 Fixed-variance/MSE 行，与 LSTM 侧
MC_LSTM 的角色完全对应。
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
    """逐字复用 E3_run_save_ece.py 第170-196行。"""
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
    """HeteroscedasticTransformer 去掉 sigma 头的单输出 MSE 版本，与 MC_LSTM
    （mc_dropout_model.py）角色对应：用于 MC-Dropout 测试时采样和
    Fixed-variance/MSE 行。"""
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
    """backbone-dispatching checkpoint loader，用于 NLL(Heteroscedastic*)模型。
    LSTM 分支直接复用 common.load_checkpoint_model；Transformer 分支
    额外需要 seq_len（固定=SEQUENCE_LENGTH，两骨干共用同一窗口长度）。"""
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
    """backbone-dispatching checkpoint loader，用于 MC-Dropout/MSE(MC_*)模型。"""
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
        # 复用已有的 leakfree LSTM checkpoint（不重训）
        d = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
        return os.path.join(d, f"{ds}_LSTM_{tag}{seed}.pt")
    return os.path.join(T2_CKPT_DIR, f"{ds}_Transformer_{tag}{seed}.pt")


def mc_ckpt_path(backbone, ds, seed):
    if backbone == 'LSTM':
        d = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
        return os.path.join(d, f"{ds}_MCDropoutMSE_seed{seed}.pt")
    return os.path.join(T2_CKPT_DIR, f"{ds}_MCDropoutMSE_Transformer_seed{seed}.pt")
