import torch
import torchvision
import argparse
from tqdm.auto import tqdm
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import Adam
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import setup_seed, count_parameters
from model import BarlowTwins 
from config import CFG 
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train Barlow Twins on CIFAR-10")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training and validation")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory to save checkpoints")
    parser.add_argument('--save_epoch', type=int, default=5, help="Save model every n epochs")
    parser.add_argument('--path_to_encoder_model', type=str, help="Path to the pretrained Barlow Twins model")

    return parser.parse_args()

class Transform:
    def __init__(self):
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x)

if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    config = CFG()
    wandb.init(project="barlow_twins_cifar10_training", config=args, group="BarlowTwins")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Barlow Twins model
    model  = BarlowTwins(config=config).to(DEVICE) 
    checkpoint = torch.load(args.path_to_encoder_model)
    missing_keys, unexpected_keys  = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print("USING THE PRE-TRAINED ENCODER")

    num_ftrs = model.encoder.fc.in_features
    model.encoder.fc = nn.Linear(num_ftrs, 10).to(DEVICE)
    model.encoder.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.encoder.fc.bias.data.zero_()

    for name, param in model.named_parameters():
        if 'encoder.fc' not in name:  
            param.requires_grad = False
        else:
            param.requires_grad = True


    
    print(count_parameters(model))

    transform = Transform()
    train_data = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    val_data = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model.encoder(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            wandb.log({
                "train_loss": loss.item(),
                "train_acc": 100. * correct / total,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
                "step": step + 1,
            })

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model.encoder(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch + 1,
        })

        print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {train_loss/len(train_loader):.4f}, Train acc: {100.*correct/total:.2f}%, Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}%")

        if (epoch + 1) % args.save_epoch == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"barlow_twins_cifar10_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
