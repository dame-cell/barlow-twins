import torch
import torchvision
import argparse
from tqdm.auto import tqdm
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from config import CFG
from model import BarlowTwins
from optimizer import Lars
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import Transform, count_parameters, setup_seed
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def warmup_lr(epoch, warmup_epochs=60):
    return min((epoch + 1) / warmup_epochs, 1.0)

def parse_args():
    parser = argparse.ArgumentParser(description="Train barlow twins on CIFAR-10 with DDP")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.2, help="Learning rate")
    parser.add_argument('--lr_bias', type=float, default=0.0048, help="Learning rate for the bias and batch norm parameters")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training and validation")
    parser.add_argument('--weight_decay', type=float, default=1.5e-6, help="Weight decay for optimizer")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--checkpoint_dir', type=str, default="data", help="Where to save your checkpoint model")
    parser.add_argument('--save_epoch', type=int, default=40, help="At what epoch to save your model")
    parser.add_argument('--world_size', type=int, default=2, help="Number of processes (GPUs) for DDP")
    parser.add_argument('--compile', action='store_true', help="Use torch.compile() to speed up training")
    return parser.parse_args()

def main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    setup_seed(args.seed + rank)  # Ensure different seeds for each process
    
    config = CFG()
    model = BarlowTwins(config=config)
    count_parameters(model)

    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    
    if args.compile:
        try:
            model = torch.compile(model)
            print(f"Model compiled successfully on rank {rank}")
        except Exception as e:
            print(f"Failed to compile model on rank {rank}: {e}")
            print("Proceeding with uncompiled model")
    
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        wandb.login(key="04098c64a0b88d5f4ff90335b7f75613041420c6")
        wandb.init(project="barlow-twins-ddp", config=args, group="DDP-Experiment")

    transform = Transform()
    train_data = torchvision.datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform)
    val_data = torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler
    )

    lr_adjusted = args.lr * args.batch_size * world_size / 256
    params = [
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'bn' not in n]},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n or 'bn' in n], 'lr': args.lr_bias, 'weight_decay': 0}
    ]

    optimizer = Lars(params, lr=lr_adjusted, momentum=0.9, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_adjusted * warmup_lr(epoch)

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}", disable=(rank != 0))
        for step, ((y1, y2), _) in progress_bar:
            y1, y2 = y1.to(device), y2.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(y1, y2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                    "step": step + 1,
                })

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, ((y1, y2), _) in enumerate(val_loader):
                y1, y2 = y1.to(device), y2.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    val_loss += model(y1, y2).item()

        val_loss /= len(val_loader)
        if rank == 0:
            wandb.log({
                "val_loss": val_loss,
                "epoch": epoch + 1,
            })
            print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {loss:.4f}, Val loss: {val_loss:.4f}")

        if (epoch + 1) % args.save_epoch == 0 and rank == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"barlow_twins_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),  # Save the underlying model
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"Model saved at {checkpoint_path}")

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
