import torch
from model.alternate.gumnet_ap import GumNet
from typing import Union


def init_gumnetap(device: Union[str, torch.device] = "cpu"):
    ckpt_path = "model/alternate/gumnetap_2d_best_noise_level_0_8x8_200.pth.zip"
    model = GumNet(grid_size=8)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    return model