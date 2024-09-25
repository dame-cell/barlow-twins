import torch 
import torchvision
import argparse 
from tqdm.auto import tqdm 
import torch.nn as nn 
import wandb
from torch.utils.data import DataLoader
from config import CFG 
from model import BarlowTwins
from bitsandbytes.optim import LARS
from optimizer import Lars 
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import Transform, count_parameters, setup_seed
import os 

def warmup_lr(epoch, warmup_epochs=10):
    return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

def parse_args():
    parser = argparse.ArgumentParser(description="Train barlow twins on CIFAR-10")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs ")
    parser.add_argument('--lr', type=float, default=0.2, help="Learning rate")
    parser.add_argument('--lr_bias', type=float, default=0.0048, help="Learning rate for the bias and the d batch normalization parameters ")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training and validation ")
    parser.add_argument('--weight_decay', type=float, default=1.5e-6 , help="Weight decay for optimizer")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility ")
    parser.add_argument('--image_path', type=str, help="Path to an image for inference visualization")
    parser.add_argument('--mask_ratio', type=float, default=0.75, help="Masking ratio for MAE ")
    parser.add_argument('--checkpoint_dir', type=str, default="data", help="where to save your checkpoint model ")
    parser.add_argument('--save_epoch', type=int, default=10, help="At what epoch to save your model")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = CFG()
    model = BarlowTwins(config=config)
    count_parameters(model)
    
    setup_seed(args.seed)
    wandb.login(key="04098c64a0b88d5f4ff90335b7f75613041420c6")
    wandb.init(project="barlow-twins-training", config=args, group=f"Barlow-twins",)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    transform = Transform()
    train_data = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    val_data = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

 

    lr_adjusted = args.lr * args.batch_size / 256

    params = [
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'bn' not in n]},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n or 'bn' in n], 'lr': args.lr_bias, 'weight_decay': 0}
    ]

    # LARS optimizer
    optimizer = Lars(params, momentum=0.9, lr=lr_adjusted, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 10, eta_min=lr_adjusted / 1000)
    scaler = torch.GradScaler()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        # Adjust learning rate for warm-up
        for param_group in optimizfer.param_groups:
            param_group['lr'] = lr_adjusted * warmup_lr(epoch)

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, ((y1,y2),_) in progress_bar:
            y1 = y1.to(DEVICE)
            y2 = y2.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(y1,y2)
             
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "epoch": epoch + 1,
                        "step": step + 1,
                    })


        model.eval()
        with torch.no_grad():
            val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation {epoch + 1}/{args.epochs}")
            val_loss = 0.0
            
            for val_step, ((y1,y2), _) in val_progress_bar:
                y1 = y1.to(DEVICE)
                y2 = y2.to(DEVICE)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    val_loss += model(y1,y2)
             

        val_loss /= len(val_loader)
        wandb.log({
                        "val_loss": val_loss,
                        "epoch": epoch + 1,
                        "step": step + 1,
                    })
        print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {loss:.4f}, Val loss: {val_loss:.4f}")

        if epoch >= 10:
            scheduler.step() 
        
        if (epoch + 1) % args.save_epoch == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"barlow_twins_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"Model saved at {checkpoint_path}")