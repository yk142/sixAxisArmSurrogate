"""#30: #24の分析(容量不足ではなくデータ不足)を検証するため、LightweightGrayBoxModel
(#19)を学習軌道数4倍(150->600)・他は既定のレシピ(アーキテクチャ・LR・カリキュラム)
のまま学習する。
"""
import numpy as np
import torch

import src.train as train_mod
from src.model import LightweightGrayBoxModel

OUT_PATH = "outputs/model_lightweight_moredata.pt"


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    train_mod.N_TRAIN_TRAJ = 600
    train_mod.N_VAL_TRAJ = 120

    model = train_mod.train(model_cls=LightweightGrayBoxModel)
    torch.save(model.state_dict(), OUT_PATH)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
