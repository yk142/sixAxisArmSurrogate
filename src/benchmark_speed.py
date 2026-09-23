"""#14, #16: 真値シミュレータ(numpy RNEA)とサロゲートモデルのロールアウト速度を
比較する。

グレーボックスモデル(src/model.py)は質量行列・バイアス力を「既知の物理」として
扱い、RNEAをtorchで再実装しているため、素朴には真値シミュレータと本質的に同じ
計算グラフ(6リンク分のPythonループを含むRNEAの前進・後退再帰)を持ち、計算量的な
高速化は期待できない(#14で確認)。`torch.compile`でRNEAのPythonループの
オーバーヘッドを削減することで、物理構造・学習済み摩擦係数を変えずに高速化できる
か検証する(#16)。初回コンパイルには数十秒のオーバーヘッドがあるため、
ウォームアップ後の定常速度と分けて計測する。

一方、ブラックボックスモデル(NSSModel、純粋なMLP)はRNEAの反復計算を持たない
ため、コンパイルなしでも大幅に高速(参考として比較する)。
"""
import time

import numpy as np
import torch

from src.model import NSSModel, StructuredFrictionGrayBoxModel
from src.physics import N_JOINTS, simulate, simulate_batch

DT = 0.002
SEED = 0


def benchmark_single_trajectory(n_steps: int = 500) -> None:
    rng = np.random.default_rng(SEED)
    ic = rng.uniform(-1, 1, size=2 * N_JOINTS)
    tau = rng.uniform(-1, 1, size=(n_steps, N_JOINTS))

    graybox = StructuredFrictionGrayBoxModel()
    graybox.eval()

    t0 = time.time()
    simulate(ic, DT, n_steps, tau=tau)
    t_true = time.time() - t0

    t0 = time.time()
    graybox.rollout(ic, n_steps, tau_seq=tau)
    t_gray = time.time() - t0

    print(f"[single trajectory, {n_steps} steps]")
    print(f"  true physics (numpy):     {t_true:.3f}s")
    print(f"  graybox surrogate(torch): {t_gray:.3f}s  (speedup {t_true / t_gray:.2f}x)")


def benchmark_batch(batch: int = 50, n_steps: int = 200) -> None:
    rng = np.random.default_rng(SEED)
    ics = rng.uniform(-1, 1, size=(batch, 2 * N_JOINTS))
    tau = rng.uniform(-1, 1, size=(batch, n_steps, N_JOINTS))
    tau_for_model = np.transpose(tau, (1, 0, 2))  # (n_steps, batch, 6)

    graybox = StructuredFrictionGrayBoxModel()
    graybox.eval()
    blackbox = NSSModel()
    blackbox.eval()

    t0 = time.time()
    simulate_batch(ics, DT, n_steps, tau)
    t_true = time.time() - t0

    t0 = time.time()
    graybox.rollout(ics, n_steps, tau_seq=tau_for_model)
    t_gray = time.time() - t0

    t0 = time.time()
    blackbox.rollout(ics, n_steps, tau_seq=tau_for_model)
    t_black = time.time() - t0

    print(f"[batch={batch}, {n_steps} steps]")
    print(f"  true physics (numpy, simulate_batch):  {t_true:.3f}s")
    print(f"  graybox surrogate (torch RNEA):         {t_gray:.3f}s  (speedup {t_true / t_gray:.2f}x)")
    print(f"  blackbox surrogate (MLP, 参考):         {t_black:.3f}s  (speedup {t_true / t_black:.2f}x)")

    # torch.compile版は形状ごとに再コンパイルが必要なため、同じ(batch, n_steps)で
    # 比較する。
    graybox_compiled = StructuredFrictionGrayBoxModel()
    graybox_compiled.load_state_dict(graybox.state_dict())
    graybox_compiled.eval()

    t0 = time.time()
    graybox_compiled.rollout(ics, n_steps, tau_seq=tau_for_model, use_compile=True)
    t_gray_compiled_with_warmup = time.time() - t0

    t0 = time.time()
    graybox_compiled.rollout(ics, n_steps, tau_seq=tau_for_model, use_compile=True)
    t_gray_compiled_steady = time.time() - t0

    print(f"  graybox + torch.compile (初回、コンパイル込み): {t_gray_compiled_with_warmup:.3f}s")
    print(
        f"  graybox + torch.compile (定常状態):             {t_gray_compiled_steady:.3f}s  "
        f"(speedup {t_true / t_gray_compiled_steady:.2f}x)"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    benchmark_single_trajectory()
    print()
    benchmark_batch()
