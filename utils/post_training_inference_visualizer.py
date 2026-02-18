import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

from model.gumnet import GumNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/gumnet_2d_best.pth'

def load_and_preprocess_image(image_path):
    transform = T.Compose([
        T.Grayscale(),
        T.Resize((192, 192)),
        T.ToTensor(),
        T.RandomInvert(p=1.0) 
    ])
    img = Image.open(image_path)
    tensor = transform(img).unsqueeze(0)
    
    return tensor.to(DEVICE)

def plot_results(template, impression, warped_impression):
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    
    axes[0].imshow(template, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Template (Target)")
    axes[0].axis('off')
    
    axes[1].imshow(impression, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Impression (Before)")
    axes[1].axis('off')
    
    axes[2].imshow(warped_impression, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Warped Impression (After)")
    axes[2].axis('off')
    
    overlay_initial = torch.zeros((192, 192, 3))
    overlay_initial[:, :, 0] = torch.tensor(template)
    overlay_initial[:, :, 1] = torch.tensor(impression)
    
    axes[3].imshow(overlay_initial.numpy())
    axes[3].set_title("Initial Overlay\n(Red=Target, Green=Before)")
    axes[3].axis('off')

    overlay_deformation = torch.zeros((192, 192, 3))
    overlay_deformation[:, :, 0] = torch.tensor(impression)
    overlay_deformation[:, :, 1] = torch.tensor(warped_impression)
    
    axes[4].imshow(overlay_deformation.numpy())
    axes[4].set_title("Deformation Overlay\n(Red=Before, Green=After)")
    axes[4].axis('off')

    overlay_final = torch.zeros((192, 192, 3))
    overlay_final[:, :, 0] = torch.tensor(template)
    overlay_final[:, :, 1] = torch.tensor(warped_impression)
    
    axes[5].imshow(overlay_final.numpy())
    axes[5].set_title("Final Alignment\n(Red=Target, Green=After)")
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()

def main(template_path, impression_path):
    
    model = GumNet(in_channels=1).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Successfully loaded weights from {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Running with untrained weights!")
        
    model.eval()
    Sa = load_and_preprocess_image(template_path)
    Sb = load_and_preprocess_image(impression_path)
    
    with torch.no_grad():
        warped_Sb, _ = model(Sa, Sb)
        
    Sa_plot = Sa.squeeze().cpu().numpy()
    Sb_plot = Sb.squeeze().cpu().numpy()
    warped_Sb_plot = warped_Sb.squeeze().cpu().numpy()
    plot_results(Sa_plot, Sb_plot, warped_Sb_plot)

if __name__ == '__main__':
    TEMPLATE_FILE = 'data/Natural/master/1.png' 
    IMPRESSION_FILE = 'data/Natural/Noise_Level_0/test/1_805.png'
    
    main(TEMPLATE_FILE, IMPRESSION_FILE)