import timm 
import torch 
import torchvision
import torch.nn as nn 
from utils import  count_parameters
from config import CFG 


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.mlp_input, self.config.mlp_output, bias=False)
        self.bn1 = nn.BatchNorm1d(self.config.mlp_output)
        self.fc2 = nn.Linear(self.config.mlp_output, self.config.mlp_output, bias=False)
        self.bn2 = nn.BatchNorm1d(self.config.mlp_output)
        self.fc3 = nn.Linear(self.config.mlp_output, self.config.mlp_output, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc3(x)
        return x


class BarlowTwins(nn.Module):
    def __init__(self, config):
        super(BarlowTwins, self).__init__()
        self.config = config
        self.encoder = timm.create_model(self.config.model_name, pretrained=self.config.pretrained, num_classes=self.config.num_classes)
        self.projector = MLP(self.config)
        
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(self.config.mlp_output, affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.encoder(y1))
        z2 = self.projector(self.encoder(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # divide by batch size (use self.config instead of self.args)
        c.div_(self.config.batch_size)

        # on-diagonal and off-diagonal loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.config.lambd * off_diag
        return loss

if __name__ == "__main__":
    config =CFG()
    model = BarlowTwins(config=config)
    count_parameters(model)
