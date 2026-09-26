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
# #8: LR=1e-3固定だと105エポック学習しても摩擦係数(特に手首関節)が初期値
# (全関節1.0)付近から動き切らず、真の係数(手動で設定すると学習損失が
# ほぼ0になることを確認済み)に到達しなかった。LR=0.05固定に上げると序盤は
# 急速に改善する(20エポックで損失が半分未満になる)ものの、最適点付近で
# 減衰がないため発振・オーバーシュートしてしまう。コサイン減衰で高LRから
# 徐々に下げることで、序盤の速い収束と終盤の安定した微調整を両立する。
LR = 1e-2
LR_MIN = 1e-4
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
    model: AutoregressiveModel | None = None,
) -> AutoregressiveModel:
    """model_cls: 新規モデルを構築する場合のクラス(既定)。model: 既に構築済みの
    モデル(#32のように事前学習済み重みの一部を凍結して続きを学習する場合)を
    渡すとmodel_clsは無視される。いずれの場合もoptimizerはrequires_grad=Trueの
    パラメータのみを更新する。
    """
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

    model = (model if model is not None else model_cls()).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    total_epochs = sum(n_epochs for _, n_epochs in curriculum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=LR_MIN)

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

            scheduler.step()
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
