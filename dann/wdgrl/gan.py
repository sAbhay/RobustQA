import torch
from torch.nn import functional as F


def loss_nonsaturating_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR DISCRIMINATOR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    zeros = torch.zeros(batch_size)
    d_loss = F.binary_cross_entropy_with_logits(d(x_real), zeros+1) + F.binary_cross_entropy_with_logits(d(g(z)), zeros)
    # YOUR CODE ENDS HERE

    return d_loss

def loss_nonsaturating_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR GENERATOR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.logsigmoid
    g_loss = -torch.mean(F.logsigmoid(d(g(z))))
    # YOUR CODE ENDS HERE

    return g_loss


def conditional_loss_nonsaturating_d(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR DISCRIMINATOR STARTS HERE
    zeros = torch.zeros(batch_size)
    d_loss = F.binary_cross_entropy_with_logits(d(x_real, y_real), zeros + 1) + F.binary_cross_entropy_with_logits(d(g(z, y_fake), y_fake),
                                                                                                           zeros)
    # YOUR CODE ENDS HERE

    return d_loss


def conditional_loss_nonsaturating_g(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR DISCRIMINATOR STARTS HERE
    # Just add y to everything
    g_loss = -torch.mean(F.logsigmoid(d(g(z, y_fake), y_fake)))
    # YOUR CODE ENDS HERE

    return g_loss


def loss_wasserstein_gp_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    x_fake = g(z)

    E_p_theta = torch.mean(d(x_fake), dim=0)
    E_p_data = torch.mean(d(x_real), dim=0)

    alpha = torch.rand(batch_size).reshape(-1, 1, 1, 1)
    r = alpha * x_fake + (1 - alpha) * x_real
    r_preds = d(r)
    grads = torch.autograd.grad(r_preds, r, grad_outputs=torch.ones_like(r_preds), create_graph=True)[0]
    grads = torch.reshape(grads, (r_preds.shape[0], -1))
    E_r = torch.mean((torch.norm(grads, dim=1) - 1) ** 2, dim=0)
    # print(r_preds.shape, r.shape, grads.shape, E_r.shape)
    lam = 10
    d_loss = E_p_theta - E_p_data + lam * E_r
    # YOUR CODE ENDS HERE

    return d_loss


def loss_wasserstein_gp_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): wasserstein generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    g_loss = -torch.mean(d(g(z)), dim=0)
    # YOUR CODE ENDS HERE

    return g_loss
