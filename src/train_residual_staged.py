"""Issue #38: ResidualGrayBoxModelの残差MLPを凍結し、摩擦係数だけを段階学習する。"""
import numpy as np
import torch

import src.train as train_mod
from src.model import ResidualGrayBoxModel

PRETRAINED_PATH = "outputs/model_residual_pretrained.pt"
OUT_PATH = "outputs/model_residual_staged.pt"


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    model = ResidualGrayBoxModel()
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location="cpu"))
    model.mass_net.requires_grad_(False)
    model.bias_net.requires_grad_(False)

    model = train_mod.train(model=model)
    torch.save(model.state_dict(), OUT_PATH)
    print(f"saved {OUT_PATH}")
    print("learned c_viscous:", torch.exp(model.log_c_viscous).detach().numpy())
    print("learned c_coulomb:", torch.exp(model.log_c_coulomb).detach().numpy())


if __name__ == "__main__":
    main()
