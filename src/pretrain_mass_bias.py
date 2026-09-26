"""#32: LightweightGrayBoxModel(#19)のmass_net/bias_netを、ロールアウト損失
ではなく真の物理(mass_matrix/bias_forces)への直接教師あり回帰で事前学習する。

#30までの学習はM(q)・bias(q,q̇)・摩擦を同時に多段ロールアウト損失(ODE積分を
通した間接的な誤差信号)だけで学習していた。このシミュレーション研究では真の
物理が既知なので、ランダムサンプリングした(q)/(q,q̇)に対する真のmass_matrix/
bias_forcesを教師ラベルとして直接MSE回帰すれば、ロールアウトの間接性・摩擦との
干渉を完全に切り離して「MLPがこの関数を表現・学習できるか」を検証できる。
"""
import numpy as np
import torch

from src.model import LightweightGrayBoxModel
from src.physics import N_JOINTS, bias_forces, mass_matrix

DT_DEFAULT = 0.002
SEED = 0

N_TRAIN = 200_000
N_VAL = 20_000
BATCH_SIZE = 1024
N_EPOCHS = 80
LR = 1e-3

# データセットの初期速度レンジ(-1,1)より広く、ロールアウト中に到達しうる速度
# 域も含めてbias_net全体をカバーする(VELOCITY_LIMIT=15までの棄却サンプリング
# より狭いが、TAU_MAX由来で実際に頻繁に探索される範囲を優先する)。
Q_DOT_RANGE = (-4.0, 4.0)


def sample_mass_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    q = rng.uniform(-np.pi, np.pi, size=(n, N_JOINTS))
    m_true = mass_matrix(q)  # (n, 6, 6)
    return q.astype(np.float32), m_true.astype(np.float32)


def sample_bias_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = rng.uniform(-np.pi, np.pi, size=(n, N_JOINTS))
    q_dot = rng.uniform(*Q_DOT_RANGE, size=(n, N_JOINTS))
    bias_true = bias_forces(q, q_dot)
    return q.astype(np.float32), q_dot.astype(np.float32), bias_true.astype(np.float32)


def train_mass_net(model: LightweightGrayBoxModel) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    q_train, m_train = sample_mass_data(N_TRAIN, rng)
    q_val, m_val = sample_mass_data(N_VAL, rng)

    q_train_t = torch.as_tensor(q_train)
    m_train_t = torch.as_tensor(m_train)
    q_val_t = torch.as_tensor(q_val)
    m_val_t = torch.as_tensor(m_val)

    # #32での検証: bias(q,q̇)と違いM(q)の成分間スケール差は1.6-26%相対誤差
    # 程度と軽微(手首関節でもM(q)の絶対値自体は非ゼロで無視できるほど小さく
    # ない)。むしろ一部の成分(例: M11)は分散が非常に小さい(ほぼ定数)ため、
    # 標準偏差で正規化すると些細な絶対誤差が正規化後に爆発し、他の成分の
    # 学習を阻害する数値不安定を招くことを確認した。そのためM(q)は正規化なし
    # の生MSEのまま学習する(bias_netとは対照的、下記参照)。
    optimizer = torch.optim.Adam(model.mass_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)
    n = q_train_t.shape[0]

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            pred = model.mass_matrix(q_train_t[idx])
            loss = ((pred - m_train_t[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.shape[0]
        epoch_loss /= n
        scheduler.step()

        with torch.no_grad():
            val_pred = model.mass_matrix(q_val_t)
            val_loss = ((val_pred - m_val_t) ** 2).mean().item()
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(f"[mass_net] epoch {epoch+1:3d} train_mse={epoch_loss:.6f} val_mse={val_loss:.6f}")

    with torch.no_grad():
        val_pred = model.mass_matrix(q_val_t)
        rel_err = (val_pred - m_val_t).norm() / m_val_t.norm()
    print(f"[mass_net] final val relative error = {rel_err.item():.4f}")
    return epoch_loss, val_loss


def train_bias_net(model: LightweightGrayBoxModel) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + 1)
    q_train, qd_train, bias_train = sample_bias_data(N_TRAIN, rng)
    q_val, qd_val, bias_val = sample_bias_data(N_VAL, rng)

    g_train = bias_forces(q_train, np.zeros_like(qd_train)).astype(np.float32)
    c_train = (bias_train - g_train).astype(np.float32)
    g_val = bias_forces(q_val, np.zeros_like(qd_val)).astype(np.float32)
    c_val = (bias_val - g_val).astype(np.float32)

    state_train = torch.as_tensor(np.concatenate([q_train, qd_train], axis=-1))
    g_train_t = torch.as_tensor(g_train)
    c_train_t = torch.as_tensor(c_train)
    bias_train_t = g_train_t + c_train_t
    state_val = torch.as_tensor(np.concatenate([q_val, qd_val], axis=-1))
    g_val_t = torch.as_tensor(g_val)
    c_val_t = torch.as_tensor(c_val)
    bias_val_t = g_val_t + c_val_t

    from src.model import encode_state

    # bias(q,q̇)は腕関節(q1-3)と手首関節(q4-6)で振幅が1-600倍異なる。
    # 分解後も各物理項の成分ごとの標準偏差で正規化し、振幅差による学習
    # トレードオフを避ける。
    g_channel_std = g_train_t.std(dim=0, keepdim=True).clamp_min(1e-4)
    c_channel_std = c_train_t.std(dim=0, keepdim=True).clamp_min(1e-4)
    bias_channel_std = bias_train_t.std(dim=0, keepdim=True).clamp_min(1e-4)

    optimizer = torch.optim.Adam(model.bias_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)
    n = state_train.shape[0]

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            enc = encode_state(state_train[idx])
            g_pred = model.bias_net.gravity(enc)
            c_pred = model.bias_net.coriolis(enc)
            g_loss = (((g_pred - g_train_t[idx]) / g_channel_std) ** 2).mean()
            c_loss = (((c_pred - c_train_t[idx]) / c_channel_std) ** 2).mean()
            loss = g_loss + c_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.shape[0]
        epoch_loss /= n
        scheduler.step()

        with torch.no_grad():
            enc_val = encode_state(state_val)
            g_val_pred = model.bias_net.gravity(enc_val)
            c_val_pred = model.bias_net.coriolis(enc_val)
            val_g_loss = (((g_val_pred - g_val_t) / g_channel_std) ** 2).mean()
            val_c_loss = (((c_val_pred - c_val_t) / c_channel_std) ** 2).mean()
            val_loss = (val_g_loss + val_c_loss).item()
            val_pred = g_val_pred + c_val_pred
            val_bias_loss = (((val_pred - bias_val_t) / bias_channel_std) ** 2).mean().item()
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(
                f"[bias_net] epoch {epoch+1:3d} train_mse={epoch_loss:.6f} "
                f"val_mse={val_loss:.6f} val_g={val_g_loss.item():.6f} "
                f"val_c={val_c_loss.item():.6f} val_bias={val_bias_loss:.6f}"
            )

    with torch.no_grad():
        enc_val = encode_state(state_val)
        g_val_pred = model.bias_net.gravity(enc_val)
        c_val_pred = model.bias_net.coriolis(enc_val)
        val_pred = g_val_pred + c_val_pred
        rel_err = (val_pred - bias_val_t).norm() / bias_val_t.norm()
        g_rel_joint = (g_val_pred - g_val_t).norm(dim=0) / g_val_t.norm(dim=0).clamp_min(1e-8)
        c_rel_joint = (c_val_pred - c_val_t).norm(dim=0) / c_val_t.norm(dim=0).clamp_min(1e-8)
    print(f"[bias_net] final val relative error = {rel_err.item():.4f}")
    print(
        "[bias_net] gravity relative error by joint = "
        + ", ".join(f"q{i+1}={100.0 * e.item():.2f}%" for i, e in enumerate(g_rel_joint))
    )
    print(
        "[bias_net] coriolis relative error by joint = "
        + ", ".join(f"q{i+1}={100.0 * e.item():.2f}%" for i, e in enumerate(c_rel_joint))
    )
    return epoch_loss, val_loss


def main() -> None:
    torch.manual_seed(0)
    model = LightweightGrayBoxModel()

    print("=== mass_net direct supervised pretraining ===")
    train_mass_net(model)

    print("=== bias_net direct supervised pretraining ===")
    train_bias_net(model)

    torch.save(model.state_dict(), "outputs/model_lightweight_pretrained_massbias.pt")
    print("saved outputs/model_lightweight_pretrained_massbias.pt")


if __name__ == "__main__":
    main()
