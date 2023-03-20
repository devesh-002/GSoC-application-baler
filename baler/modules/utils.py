import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple

# import torch.tensor as Tensor
# import modules.helper as helper

# import torch.tensor as Tensor
Tensor = TypeVar('torch.tensor')
# config, mode, project_name = helper.get_arguments()

###############################################
factor = 0.5
min_lr = 1e-6


###############################################


def new_loss_func(model, reconstructed_data, true_data, reg_param, val):
    # Still WIP. Other loss function is completely fine!
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)
    l1_loss = 0

    if not val:
        for i in model.parameters():
            l1_loss += torch.abs(i).sum()

            # l1_loss = sum(torch.sum(torch.abs(p)) for p in model.parameters())

            loss = mse_loss + reg_param * l1_loss
            return loss, mse_loss, l1_loss

    else:
        loss = mse_loss
        return loss
    
def sparse_loss_function_L1(
    model_children, true_data, reconstructed_data, reg_param, validate
):
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)

    l1_loss = 0
    values = true_data
    if not validate:
        for i in range(len(model_children)):
            values = F.relu(model_children[i](values))
            l1_loss += torch.mean(torch.abs(values))

        loss = mse_loss + reg_param * l1_loss
        return loss, mse_loss, l1_loss
    else:
        return mse_loss

def vae_loss_function(model_children,true_data,reconstructed_data,validate,encode,reg_param,reg_weigh: int = 100) -> dict:
    # recons = args[0]
    # input = args[1]
    # z = args[2]
    z=encode
    batch_size = 512
    bias_corr = batch_size * (batch_size - 1)
    reg_weight = reg_weigh / bias_corr

    recons_loss = F.mse_loss(reconstructed_data, true_data)

    mmd_loss = compute_mmd(z, reg_weight)

    loss = recons_loss + mmd_loss
    return  loss,  recons_loss, mmd_loss

def compute_kernel(
                    x1: Tensor,
                    x2: Tensor,kernel_type='imq') -> Tensor:
    # Convert the tensors into row and column vectors
    D = x1.size(1)
    N = x1.size(0)

    x1 = x1.unsqueeze(-2)  # Make it into a column tensor
    x2 = x2.unsqueeze(-3)  # Make it into a row tensor

    """
    Usually the below lines are not required, especially in our case,
    but this is useful when x1 and x2 have different sizes
    along the 0th dimension.
    """
    x1 = x1.expand(N, N, D)
    x2 = x2.expand(N, N, D)

    if kernel_type == 'rbf':
        result = compute_rbf(x1, x2)
    elif kernel_type == 'imq':
        result = compute_inv_mult_quad(x1, x2)
    else:
        raise ValueError('Undefined kernel type.')

    return result

def compute_rbf(
                x1: Tensor,
                x2: Tensor,
                eps: float = 1e-7,latent_var=2.0) -> Tensor:
    """
    Computes the RBF Kernel between x1 and x2.
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    sigma = 2. * z_dim * latent_var

    result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
    return result

def compute_inv_mult_quad(
                            x1: Tensor,
                            x2: Tensor,
                            eps: float = 1e-7,latent_var=2.0) -> Tensor:
    """
    Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
    given by

            k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    C = 2 * z_dim * latent_var
    kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

    # Exclude diagonal elements
    result = kernel.sum() - kernel.diag().sum()

    return result

def compute_mmd( z: Tensor, reg_weight: float) -> Tensor:
    # Sample from prior (Gaussian) distribution
    prior_z = torch.randn_like(z)

    prior_z__kernel = compute_kernel(prior_z, prior_z)
    z__kernel = compute_kernel(z, z)
    priorz_z__kernel = compute_kernel(prior_z, z)

    mmd = reg_weight * prior_z__kernel.mean() + \
        reg_weight * z__kernel.mean() - \
        2 * reg_weight * priorz_z__kernel.mean()
    return mmd

def swae_loss_function(model_children,            true_data,
            reconstructed_data,
            encode,
            reg_param,z_dim
,p=2.0,reg_weight=100,validate=False) -> dict:
    recons = true_data
    input = reconstructed_data
    z = encode

    batch_size = input.size(0)
    bias_corr = batch_size *  (batch_size - 1)
    reg_weight = reg_weight / bias_corr

    recons_loss_l2 = F.mse_loss(recons, input)
    recons_loss_l1 = F.l1_loss(recons, input)

    swd_loss = compute_swd(z,  reg_weight,z_dim)

    loss = recons_loss_l2 + recons_loss_l1 + swd_loss
    return  loss, recons_loss_l2 + recons_loss_l1,  swd_loss

def get_random_projections(latent_dim: int, num_samples: int,proj_dist='normal') -> Tensor:

    if proj_dist == 'normal':
        rand_samples = torch.randn(num_samples, latent_dim)
    elif proj_dist == 'cauchy':
        rand_samples = dist.Cauchy(torch.tensor([0.0]),
                                    torch.tensor([1.0])).sample((num_samples, latent_dim)).squeeze()
    else:
        raise ValueError('Unknown projection distribution.')

    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1,1)
    return rand_proj # [S x D]


def compute_swd(   z: Tensor,
                reg_weight,z_dim,num_projections=50,p=2) -> Tensor:

    prior_z = torch.randn_like(z) # [N x D]
    device = z.device

    proj_matrix = get_random_projections(z_dim,
                                                num_samples=num_projections).transpose(0,1).to(device)

    latent_projections = z.matmul(proj_matrix.double()) # [N x S]
    prior_projections = prior_z.matmul(proj_matrix.double()) # [N x S]

    # The Wasserstein distance is computed by sorting the two projections
    # across the batches and computing their element-wise l2 distance
    w_dist = torch.sort(latent_projections.t(), dim=1)[0] - \
                torch.sort(prior_projections.t(), dim=1)[0]
    w_dist = w_dist.pow(p)
    return reg_weight * w_dist.mean()


# def binary_loss(
#     model_children, true_data, reconstructed_data, reg_param, validate
#                 ):
    

# Accuracy function still WIP. Not working properly.
# Probably has to do with total_correct counter.


def accuracy(model, dataloader):
    print("Accuracy")
    model.eval()

    total_correct = 0
    total_instances = 0

    with torch.no_grad():
        for data in tqdm(dataloader):
            x, _ = data
            classifications = torch.argmax(x)

            correct_pred = torch.sum(classifications == x).item()

            total_correct += correct_pred
            total_instances += len(x)

    accuracy_frac = round(total_correct / total_instances, 3)
    print(accuracy_frac)
    return accuracy_frac


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience  # Nr of times we allow val. loss to not improve before early stopping
        self.min_delta = min_delta  # min(new loss - best loss) for new loss to be considered improvement
        self.counter = 0  # counts nr of times val_loss dosent improve
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  ## Resets if val_loss improves

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1

            print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("Early Stopping")
                self.early_stop = True


class LRScheduler:
    def __init__(self, optimizer, patience, min_lr=min_lr, factor=factor):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        # Maybe add if statements for selectment of lr schedulers
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
