import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from fingernet import FingerNet
from fingernet_wrapper import FingerNetWrapper
import sys

# Ensure project root is on sys.path when running this file directly
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.crop_resize import crop_resize_horizontal


# Resolve paths relative to this script so the script works from any CWD
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_IMAGE_PATH = os.path.abspath(
    os.path.join(
        _SCRIPT_DIR,
        '..',
        'data',
        'FCV',
        'FVC2004',
        'Dbs',
        'DB1_A',
        '1_1.tif',
    )
)
_MODEL_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, 'fingernet.pth'))

_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(padding=(62, 0, 63, 0), fill=255),
    #transforms.Resize((192, 192)),
    transforms.RandomInvert(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

im = Image.open(_IMAGE_PATH).convert("L")

# Convert image to tensor, apply transforms, and add batch dimension
im_tensor = _TRANSFORM(im).unsqueeze(0)  # type: ignore # Shape: (1, 1, H, W)
im_tensor = -im_tensor
im_tensor = crop_resize_horizontal(im_tensor, output_size=(640, 480))

def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

def _pad_to_patch_multiple(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, _, h, w = x.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

def test1():
    model = FingerNet()
    model.load_state_dict(torch.load(_MODEL_PATH, map_location='cpu'))
    model.eval()
    model.to('cpu')
    with torch.no_grad():
        padded = _pad_to_multiple(im_tensor, 8)
        output = model(padded)
    print("FingerNet ran successfully")

def test2():
    model = FingerNet()
    model.load_state_dict(torch.load(_MODEL_PATH, map_location='cpu'))
    model.eval()
    model.to('cpu')

    wrapper = FingerNetWrapper(model)
    wrapper.to('cpu')

    out = wrapper(im_tensor)

    wrapper.plot_minutiae(im_tensor, out, save_path='minutiae_detection.png')

def test3(patch_size: int = 800):
    model = FingerNet()
    model.load_state_dict(torch.load(_MODEL_PATH, map_location='cpu'))
    model.eval()
    model.to('cpu')

    padded = _pad_to_patch_multiple(im_tensor, patch_size)
    _, c, h, w = padded.shape

    patches = F.unfold(padded, kernel_size=patch_size, stride=patch_size)
    patches = patches.transpose(1, 2).reshape(-1, c, patch_size, patch_size)

    with torch.no_grad():
        padded_patches = _pad_to_multiple(patches, 8)
        _ = model(padded_patches)

    print("FingerNet ran successfully on batch")


if __name__ == '__main__':
    test2()
    test1()
    test3()