import torch
from torch.nn import functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, dim_z=64, num_channels=1):
        super().__init__()
        self.dim_z = dim_z
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_z, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_z, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_z, 512),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, z):
        return self.net(z)


bce = torch.nn.BCELoss()
def d_loss(pred, target):
        return bce(pred, target)


D_KL = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
def D_KL_uniform(probs):
    log_probs = torch.log(probs)
    uniform = torch.ones(log_probs)
    uniform /= torch.sum(uniform, dim=-1)
    return D_KL(probs, uniform)
