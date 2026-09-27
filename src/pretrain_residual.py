"""Issue #38: ResidualGrayBoxModelのM/bias残差を真値物理へ直接教師あり事前学習する。"""
import os

import numpy as np
import torch

from src.model import ResidualGrayBoxModel
from src.physics import N_JOINTS, bias_forces, mass_matrix

SEED = 0
N_TRAIN = int(os.environ.get("M38_N_TRAIN", "200000"))
N_VAL = int(os.environ.get("M38_N_VAL", "20000"))
BATCH_SIZE = int(os.environ.get("M38_BATCH_SIZE", "1024"))
N_EPOCHS = int(os.environ.get("M38_N_EPOCHS", "80"))
LR = float(os.environ.get("M38_LR", "0.001"))
Q_DOT_RANGE = (-4.0, 4.0)
OUT_PATH = "outputs/model_residual_pretrained.pt"


def sample_mass_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    q = rng.uniform(-np.pi, np.pi, size=(n, N_JOINTS))
    return q.astype(np.float32), mass_matrix(q).astype(np.float32)


def sample_bias_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = rng.uniform(-np.pi, np.pi, size=(n, N_JOINTS))
    q_dot = rng.uniform(*Q_DOT_RANGE, size=(n, N_JOINTS))
    return q.astype(np.float32), q_dot.astype(np.float32), bias_forces(q, q_dot).astype(np.float32)


def rms_scale(target: torch.Tensor, reduce_dims: tuple[int, ...]) -> torch.Tensor:
    rms = torch.sqrt((target**2).mean(dim=reduce_dims, keepdim=True))
    floor = torch.median(rms.detach().flatten()).clamp_min(1e-8) * 0.01
    return rms.clamp_min(floor)


def format_channel_errors(name: str, pred: torch.Tensor, target: torch.Tensor) -> None:
    err_rms = torch.sqrt(((pred - target) ** 2).mean(dim=0))
    true_rms = torch.sqrt((target**2).mean(dim=0))
    floor = torch.median(true_rms).clamp_min(1e-8) * 0.01
    parts = []
    for i in range(err_rms.numel()):
        if true_rms[i] < floor:
            parts.append(f"j{i+1}=abs {err_rms[i].item():.4g}")
        else:
            parts.append(f"j{i+1}=rel {(err_rms[i] / true_rms[i]).item():.4f}")
    print(f"[{name}] " + "  ".join(parts))


def evaluate(model: ResidualGrayBoxModel, q_val_t: torch.Tensor, m_val_t: torch.Tensor, state_val: torch.Tensor, bias_val_t: torch.Tensor) -> None:
    with torch.no_grad():
        base_m = model.base_mass_matrix(q_val_t)
        pred_m = model.mass_matrix(q_val_t)
        base_bias = model.base_bias(state_val[..., :N_JOINTS], state_val[..., N_JOINTS:])
        pred_bias = model.bias(state_val)

    print("=== channel errors: M diagonal ===")
    diag_true = torch.diagonal(m_val_t, dim1=-2, dim2=-1)
    format_channel_errors("base Mdiag", torch.diagonal(base_m, dim1=-2, dim2=-1), diag_true)
    format_channel_errors("res  Mdiag", torch.diagonal(pred_m, dim1=-2, dim2=-1), diag_true)
    print("=== channel errors: bias ===")
    format_channel_errors("base bias", base_bias, bias_val_t)
    format_channel_errors("res  bias", pred_bias, bias_val_t)


def train_mass_net(model: ResidualGrayBoxModel, q_train_t: torch.Tensor, m_train_t: torch.Tensor, q_val_t: torch.Tensor, m_val_t: torch.Tensor) -> None:
    scale = rms_scale(m_train_t, reduce_dims=(0,))
    optimizer = torch.optim.Adam(model.mass_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)
    n = q_train_t.shape[0]
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            pred = model.mass_matrix(q_train_t[idx])
            loss = (((pred - m_train_t[idx]) / scale) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.shape[0]
        scheduler.step()
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            with torch.no_grad():
                val_loss = (((model.mass_matrix(q_val_t) - m_val_t) / scale) ** 2).mean().item()
            print(f"[mass_residual] epoch {epoch+1:3d} train_mse={epoch_loss / n:.6f} val_mse={val_loss:.6f}")


def train_bias_net(
    model: ResidualGrayBoxModel,
    state_train: torch.Tensor,
    bias_train_t: torch.Tensor,
    state_val: torch.Tensor,
    bias_val_t: torch.Tensor,
) -> None:
    scale = rms_scale(bias_train_t, reduce_dims=(0,))
    optimizer = torch.optim.Adam(model.bias_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)
    n = state_train.shape[0]
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            pred = model.bias(state_train[idx])
            loss = (((pred - bias_train_t[idx]) / scale) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.shape[0]
        scheduler.step()
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            with torch.no_grad():
                val_loss = (((model.bias(state_val) - bias_val_t) / scale) ** 2).mean().item()
            print(f"[bias_residual] epoch {epoch+1:3d} train_mse={epoch_loss / n:.6f} val_mse={val_loss:.6f}")


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs("outputs", exist_ok=True)
    model = ResidualGrayBoxModel()

    q_train, m_train = sample_mass_data(N_TRAIN, rng)
    q_val, m_val = sample_mass_data(N_VAL, rng)
    q_b_train, qd_train, bias_train = sample_bias_data(N_TRAIN, rng)
    q_b_val, qd_val, bias_val = sample_bias_data(N_VAL, rng)

    q_train_t = torch.as_tensor(q_train)
    m_train_t = torch.as_tensor(m_train)
    q_val_t = torch.as_tensor(q_val)
    m_val_t = torch.as_tensor(m_val)
    state_train = torch.as_tensor(np.concatenate([q_b_train, qd_train], axis=-1))
    bias_train_t = torch.as_tensor(bias_train)
    state_val = torch.as_tensor(np.concatenate([q_b_val, qd_val], axis=-1))
    bias_val_t = torch.as_tensor(bias_val)

    print("=== before residual pretraining ===")
    evaluate(model, q_val_t, m_val_t, state_val, bias_val_t)
    print("=== mass residual supervised pretraining ===")
    train_mass_net(model, q_train_t, m_train_t, q_val_t, m_val_t)
    print("=== bias residual supervised pretraining ===")
    train_bias_net(model, state_train, bias_train_t, state_val, bias_val_t)
    print("=== after residual pretraining ===")
    evaluate(model, q_val_t, m_val_t, state_val, bias_val_t)

    torch.save(model.state_dict(), OUT_PATH)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
