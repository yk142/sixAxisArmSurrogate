"""6軸アーム用のNSSモデル学習スクリプト。

学習ループはscaraSurrogate Phase1-3と同じレシピ(マルチステップ損失+
カリキュラム学習)を用い、損失は成分ごとに正規化する(Phase2-M8の教訓)。
"""
import numpy as np
import torch
import torch.nn as nn

from src.dataset import generate_controlled_trajectories, make_rollout_windows
from src.model import AutoregressiveModel, StructuredFrictionGrayBoxModel, encode_state

DT = 0.002
N_STEPS_PER_TRAJ = 60
N_TRAIN_TRAJ = 150
N_VAL_TRAJ = 30
SEED = 0
BATCH_SIZE = 256
LR = 1e-3
GRAD_CLIP_NORM = 1.0

# torch版RNEAはmass_matrix 1回でも内部で6リンク分のPythonループを2周する
# ため、scaraSurrogate Phase1-3(解析的なM/C/G、K_MAX=30)より1ステップ
# あたりのコストが大きい。データセット規模・ロールアウト窓・カリキュラムの
# エポック数を全体的に縮小し、実用的な学習時間に収める(諸元自体は今回も
# 厳密さより「妥当な範囲で適当」を許容する方針)。
K_MAX = 10
WINDOW_STRIDE = 3

CURRICULUM = [(1, 15), (3, 15), (5, 15), (K_MAX, 60)]


def train(
    device: str = "cpu",
    curriculum: list[tuple[int, int]] | None = None,
    model_cls: type[AutoregressiveModel] = StructuredFrictionGrayBoxModel,
) -> AutoregressiveModel:
    curriculum = curriculum if curriculum is not None else CURRICULUM
    k_max = max(k for k, _ in curriculum)

    train_traj, train_tau = generate_controlled_trajectories(N_TRAIN_TRAJ, DT, N_STEPS_PER_TRAJ, seed=SEED)
    val_traj, val_tau = generate_controlled_trajectories(N_VAL_TRAJ, DT, N_STEPS_PER_TRAJ, seed=SEED + 1)

    x0_train, u_train, targets_train = make_rollout_windows(train_traj, train_tau, k_max, stride=WINDOW_STRIDE)
    x0_val, u_val, targets_val = make_rollout_windows(val_traj, val_tau, k_max, stride=WINDOW_STRIDE)

    x0_train_t = torch.as_tensor(x0_train, dtype=torch.float32, device=device)
    u_train_t = torch.as_tensor(u_train, dtype=torch.float32, device=device)
    target_enc_train = encode_state(torch.as_tensor(targets_train, dtype=torch.float32, device=device))

    x0_val_t = torch.as_tensor(x0_val, dtype=torch.float32, device=device)
    u_val_t = torch.as_tensor(u_val, dtype=torch.float32, device=device)
    target_enc_val = encode_state(torch.as_tensor(targets_val, dtype=torch.float32, device=device))

    # Phase2-M8の教訓: 角速度成分は角度成分より分散が大きく、素のMSEでは
    # 損失が角速度誤差に支配される。学習データの標準偏差で正規化する。
    channel_std = target_enc_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-3)

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (((pred - target) / channel_std) ** 2).mean()

    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_samples = x0_train_t.shape[0]
    global_epoch = 0

    for k, n_epochs in curriculum:
        for local_epoch in range(n_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, BATCH_SIZE):
                idx = perm[i : i + BATCH_SIZE]
                pred_traj = model.rollout_diff(x0_train_t[idx], u_train_t[:k, idx], k)
                pred_enc = encode_state(pred_traj[1:])
                loss = loss_fn(pred_enc, target_enc_train[:k, idx, :])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                epoch_loss += loss.item() * idx.shape[0]

            global_epoch += 1
            if local_epoch % 10 == 0 or local_epoch == n_epochs - 1:
                with torch.no_grad():
                    val_pred_traj = model.rollout_diff(x0_val_t, u_val_t[:k], k)
                    val_pred_enc = encode_state(val_pred_traj[1:])
                    val_loss = loss_fn(val_pred_enc, target_enc_val[:k]).item()
                print(
                    f"k={k:2d} epoch {global_epoch:4d} "
                    f"train_loss={epoch_loss / n_samples:.6f} val_loss={val_loss:.6f}"
                )

    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    model = train()
    torch.save(model.state_dict(), "outputs/model_graybox.pt")
    print("saved outputs/model_graybox.pt")
    if hasattr(model, "log_c_viscous"):
        print("learned c_viscous:", torch.exp(model.log_c_viscous).detach().numpy())
        print("learned c_coulomb:", torch.exp(model.log_c_coulomb).detach().numpy())
