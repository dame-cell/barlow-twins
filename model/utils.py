import torch 
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np 
import shutil

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Format the number with commas
    formatted_params = f"{total_params:,}"
    
    print(f"Total trainable parameters: {formatted_params}")
    return total_params



def safe_download_cifar10(root='data/cifar10', train=True, transform=None):
    print(f"Attempting to load/download CIFAR-10 dataset to {root}")
    try:
        # Attempt to create the directory if it doesn't exist
        os.makedirs(root, exist_ok=True)
        
        # Check if the directory is writable
        if not os.access(root, os.W_OK):
            print(f"Warning: No write access to {root}")
        
        # Try to load or download the dataset
        dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=transform)
        print("CIFAR-10 dataset loaded successfully")
        return dataset
    except RuntimeError as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        print("Attempting to clean the directory and re-download...")
        
        # Clean the directory
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)
        
        # Try again
        try:
            dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=transform)
            print("CIFAR-10 dataset loaded successfully after cleaning")
            return dataset
        except Exception as e:
            print(f"Failed to load CIFAR-10 dataset even after cleaning: {e}")
            raise


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def warmup_lr(epoch):
    if epoch < 20:  # Warm-up for the first 10% of the epochs
        return (epoch + 1) / 20  # Linearly scale up the learning rate over 20 epochs
    else:
        return 1


# coped from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

# coped from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

# coped from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


