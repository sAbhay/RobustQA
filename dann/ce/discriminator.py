import torch
from torch.nn import functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, input_size=int(768*0.9), n_classes=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(input_size, input_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(input_size, n_classes)
        )

    def forward(self, z):
        return self.net(z)


ce = torch.nn.CrossEntropyLoss()
def d_loss(pred, target):
        return ce(pred, target)


D_KL = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
def D_KL_uniform(probs):
    log_probs = torch.log(probs)
    uniform = F.normalize(torch.ones(log_probs.shape), p=1, dim=-1).to(log_probs.get_device())
    return D_KL(probs, uniform)
