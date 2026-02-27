import torch
from evaluation.matching_model.fingernet import FingerNet
from evaluation.matching_model.fingernet_wrapper import FingerNetWrapper
from typing import Union


def init_fingernet(device: Union[str, torch.device] = "cpu"):
    device = torch.device(device)
    ckpt_path = "evaluation/matching_model/fingernet.pth"
    fingernet = FingerNet()
    state= torch.load(ckpt_path, map_location=device)
    fingernet.load_state_dict(state)
    model = FingerNetWrapper(fingernet)
    model.to(device)

    return model