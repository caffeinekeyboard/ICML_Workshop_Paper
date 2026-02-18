import os
import argparse
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from datasets.fingerprint_dataset import get_dataloaders
from model.gumnet import GumNet
from model.losses.non_linear_alignment_loss import NonLinearAlignmentLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D GumNet for Unsupervised Fingerprint Alignment")
    parser.add_argument('--data_root', type=str, default='./data', help="Path to the dataset root directory")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help="Directory to save model checkpoints")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to train on")
    return parser.parse_args()


def normalize_features(tensor):
    """
    Scales ReLU feature maps [0, inf) to [0, 1] per image in the batch.
    Crucial for SoftDiceLoss: ensures background remains 0 (sparse) while peaks hit 1.
    """
    B = tensor.size(0)
    t_max = tensor.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    return tensor / (t_max + 1e-8)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch in pbar:
        Sa = batch['Sa'].to(device)
        Sb = batch['Sb'].to(device)
        
        optimizer.zero_grad()
        
        warped_Sb, control_points = model(Sa, Sb)
        loss, dice_loss, reg_loss = criterion(warped_Sb, Sa, control_points)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * Sa.size(0)
        
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}", 
            'Dice': f"{dice_loss.item():.4f}",
            'Reg': f"{reg_loss.item():.4f}"
        })
        
    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for batch in pbar:
        Sa = batch['Sa'].to(device)
        Sb = batch['Sb'].to(device)
        
        warped_Sb, control_points = model(Sa, Sb)
        loss, dice_loss, reg_loss = criterion(warped_Sb, Sa, control_points)
        
        running_loss += loss.item() * Sa.size(0)
        
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}", 
            'Dice': f"{dice_loss.item():.4f}",
            'Reg': f"{reg_loss.item():.4f}"
        })
        
    epoch_val_loss = running_loss / len(dataloader.dataset)
    
    return epoch_val_loss



def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"--- 2D GumNet Training ---")
    print(f"Device: {args.device}")
    print(f"Data Root: {args.data_root}")
    print(f"Batch Size: {args.batch_size}, LR: {args.lr}")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        noise_levels = ['Noise_Level_0']
    )
    
    model = GumNet(in_channels=1).to(args.device)
    
    criterion = NonLinearAlignmentLoss(eps=1e-8).to(args.device)
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': args.lr},
        {'params': model.siamese_matcher.parameters(), 'lr': args.lr},
        {'params': model.spatial_aligner.parameters(), 'lr': args.lr * 0.1}
    ], weight_decay=1e-5)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': []
    }
    history_path = os.path.join(args.save_dir, 'training_history.json')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, args.device, epoch)
        val_loss = validate(model, val_loader, criterion, args.device, epoch)
    
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'gumnet_2d_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path} (Val Loss: {best_val_loss:.4f})")

    print(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}")
    print(f"Loss metrics saved to {history_path}")


if __name__ == '__main__':
    main()