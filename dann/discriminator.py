import torch
from torch.nn import functional as F
from wasserstein.layers import SinkhornDistance
from focalloss import FocalLoss
from args import get_train_test_args

class Discriminator(torch.nn.Module):
    def __init__(self, input_size=int(768*0.9), n_classes=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 768),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(768, n_classes)
        )

    def forward(self, z):
        return self.net(z)


# ce = torch.nn.CrossEntropyLoss()
# def d_loss(pred, target):
#         return ce(pred, target)

args = get_train_test_args()
gamma = args.gamma
alpha = args.alpha
def d_loss(pred, target):
    return FocalLoss(gamma=gamma, alpha=alpha)(pred, target)


D_KL = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
def D_KL_uniform(probs):
    log_probs = torch.log(probs)
    uniform = F.normalize(torch.ones(log_probs.shape), p=1, dim=-1).to(log_probs.get_device())
    return D_KL(probs, uniform)


def loss_wasserstein_gp_d(logits):
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, device=logits.get_device(), reduction=None)
    uniform = F.normalize(torch.ones(logits.shape), p=1, dim=-1).to(logits.get_device())
    dist, P, C = sinkhorn(logits, uniform)
    return dist

# def loss_wasserstein_gp_d(x_real, lam=10):
#     """
#     Arguments:
#     - g (codebase.network.Generator): The generator network
#     - d (codebase.network.Discriminator): The discriminator network
#       - Note that d outputs value of discriminator
#     - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
#     - device (torch.device): 'cpu' by default
#
#     Returns:
#     - d_loss (torch.Tensor): wasserstein discriminator loss
#     """
#     batch_size = x_real.shape[0]
#
#     x_fake = F.normalize(torch.ones(x_real.shape), p=1, dim=-1).to(x_real.get_device())
#     print(x_fake.shape, x_real.shape)
#     assert x_fake.shape == x_real.shape
#
#     E_p_theta = torch.mean(d(x_fake), dim=0)
#     E_p_data = torch.mean(d(x_real), dim=0)
#
#     alpha = torch.rand(batch_size)
#     r = alpha * x_fake + (1 - alpha) * x_real
#     r_preds = d(r)
#     grads = torch.autograd.grad(r_preds, r, grad_outputs=torch.ones_like(r_preds), create_graph=True)[0]
#     grads = torch.reshape(grads, (r_preds.shape[0], -1))
#     E_r = torch.mean((torch.norm(grads, dim=1) - 1) ** 2, dim=0)
#
#     d_loss = E_p_theta - E_p_data + lam * E_r
#
#     return d_loss
