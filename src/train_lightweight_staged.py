"""#32: 段階的カリキュラム(ユーザー提案)。src/pretrain_mass_bias.pyで直接教師
あり学習したmass_net/bias_netを凍結し、摩擦係数(12スカラー)だけを実際の
ロールアウト損失で学習する(StructuredFrictionGrayBoxModelと同じ発想)。
"""
import numpy as np
import torch

import src.train as train_mod
from src.model import LightweightGrayBoxModel

PRETRAINED_PATH = "outputs/model_lightweight_pretrained_massbias.pt"
OUT_PATH = "outputs/model_lightweight_staged.pt"


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    model = LightweightGrayBoxModel()
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location="cpu"))

    model.mass_net.requires_grad_(False)
    model.bias_net.requires_grad_(False)
    # log_c_viscous, log_c_coulomb は requires_grad=True のまま(既定)。

    model = train_mod.train(model=model)
    torch.save(model.state_dict(), OUT_PATH)
    print(f"saved {OUT_PATH}")
    print("learned c_viscous:", torch.exp(model.log_c_viscous).detach().numpy())
    print("learned c_coulomb:", torch.exp(model.log_c_coulomb).detach().numpy())


if __name__ == "__main__":
    main()
