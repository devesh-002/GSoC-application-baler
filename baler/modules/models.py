import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import distributions as dist
from torch.autograd import Variable
import numpy as np
# import torch.tensor as Tensor
# Tensor = TypeVar('torch.tensor')
cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class george_SAE(nn.Module):
    def __init__(self, device, n_features, z_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

        # encoder
        self.en1 = nn.Linear(
            n_features, 200, dtype=torch.float64, device=device)
        self.en2 = nn.Linear(200, 100, dtype=torch.float64, device=device)
        self.en3 = nn.Linear(100, 50, dtype=torch.float64, device=device)
        self.en4 = nn.Linear(50, z_dim, dtype=torch.float64, device=device)
        # decoder
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float64, device=device)
        self.de2 = nn.Linear(50, 100, dtype=torch.float64, device=device)
        self.de3 = nn.Linear(100, 200, dtype=torch.float64, device=device)
        self.de4 = nn.Linear(
            200, n_features, dtype=torch.float64, device=device)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z),z




        
class george_SAE_BN(nn.Module):
    def __init__(self,device, n_features, z_dim):
        super(george_SAE_BN, self).__init__()
        self.device=device
        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, dtype=torch.float64),
            nn.Linear(200, 100, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, dtype=torch.float64),
            nn.Linear(100, 50, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, dtype=torch.float64),
            nn.Linear(50, z_dim, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim, dtype=torch.float64),
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(50,dtype=torch.float64),
            nn.Linear(50, 100, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(100,dtype=torch.float64),
            nn.Linear(100, 200, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(200,dtype=torch.float64),
            nn.Linear(200, n_features, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            # nn.BatchNorm1d(n_features,dtype=torch.float64),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class george_SAE_Dropout_BN(nn.Module):
    def __init__(self, device,n_features, z_dim):
        super(george_SAE_Dropout_BN, self).__init__()
        self.device=device
        # encoder   
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(200,dtype=torch.float64),
            nn.Linear(200, 100, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(100,dtype=torch.float64),
            nn.Linear(100, 50, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(50,dtype=torch.float64),
            nn.Linear(50, z_dim, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(z_dim,dtype=torch.float64)
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, dtype=torch.float64),
            nn.Linear(50, 100, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, dtype=torch.float64),
            nn.Linear(100, 200, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, dtype=torch.float64),
            nn.Linear(200, n_features, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            nn.BatchNorm1d(n_features, dtype=torch.float64),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class george_SAE_Dropout(nn.Module):
    def __init__(self, device,n_features, z_dim):
        super(george_SAE_Dropout, self).__init__()
        self.device=device
        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(200, 100, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.Linear(100, 50, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(50, z_dim, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(50, 100, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(100, 200, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.Linear(200, n_features, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        # z = x.view(batch_size,a,b,c) ? What is this
        x= self.encode(x)
        return self.decode(x)

    def loss(self, model_children, true_data, reconstructed_data, reg_param):
        mse = nn.MSELoss()
        mse_loss = mse(reconstructed_data, true_data)
        l1_loss = 0
        values = true_data
        for i in range(len(model_children)):
            values = F.relu((model_children[i](values)))
            l1_loss += torch.mean(torch.abs(values))
        loss = mse_loss + reg_param * l1_loss
        return loss

# def reparameterization(mu, logvar,z_dim):
#     std = torch.exp(logvar / 2)
#     sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), z_dim))))
#     z = sampled_z * std + mu
#     return z


# class AAE(nn.Module):
# class Encoder(nn.Module):
#     def __init__(self, device, n_features, z_dim, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.z_dim=z_dim
#         self.device = device

#         # encoder
#         self.en1 = nn.Linear(
#             n_features, 200, dtype=torch.float64, device=device)
#         self.en2 = nn.Linear(200, 100, dtype=torch.float64, device=device)
#         self.en3 = nn.Linear(100, 50, dtype=torch.float64, device=device)
#         self.en4 = nn.Linear(50, z_dim, dtype=torch.float64, device=device)
#         # self.mu = nn.Linear(50, z_dim,dtype=torch.float64, device=device)
#         # self.logvar = nn.Linear(50, z_dim, dtype=torch.float64, device=device)


#     def forward(self, x):
#         h1 = F.relu(self.en1(x))
#         h2 = F.relu(self.en2(h1))
#         h3 = F.relu(self.en3(h2))
#         # mu=self.mu(h3)
#         # log=self.logvar(h3)
#         # return reparameterization(mu,log,self.z_dim)
#         return self.en4(h3)

# class Discriminator(nn.Module):
#     def __init__(self, device, n_features, z_dim, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.device = device
#         self.gn1=nn.Linear(z_dim,30, dtype=torch.float64, device=device)
#         self.gn2=nn.Linear(30,30, dtype=torch.float64, device=device)
#         self.gn3=nn.Linear(30,z_dim, dtype=torch.float64, device=device)

#     def forward(self,x):
#         g1=F.relu(self.gn1(x))
#         g2=F.relu(self.gn2(g1))
#         return self.gn3(g2)

# class Decoder(nn.Module):
#     def __init__(self, device, n_features, z_dim, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.device = device
#         self.de1 = nn.Linear(z_dim, 50, dtype=torch.float64, device=device)
#         self.de2 = nn.Linear(50, 100, dtype=torch.float64, device=device)
#         self.de3 = nn.Linear(100, 200, dtype=torch.float64, device=device)
#         self.de4 = nn.Linear(
#             200, n_features, dtype=torch.float64, device=device)

#         self.n_features = n_features
#         self.z_dim = z_dim


#     def forward(self, z):

#         h4 = F.relu(self.de1(z))
#         h5 = F.relu(self.de2(h4))
#         h6 = F.relu(self.de3(h5))
#         out = self.de4(h6)
#         return out

class Encoder(nn.Module):
    def __init__(self, device, n_features, z_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_dim=z_dim
        self.device = device

        self.en1 = nn.Linear(
            n_features, 200, dtype=torch.float64, device=device)
        self.en2 = nn.Linear(200, 100, dtype=torch.float64, device=device)
        self.en3 = nn.Linear(100, 50, dtype=torch.float64, device=device)
        # self.flat = nn.Flatten()
        self.linear = nn.Linear(50, self.z_dim,dtype=torch.float64, device=device)

    def forward(self, x):
        convolution1 = F.relu(self.en1(x))
        convolution2 = F.relu(self.en2(convolution1))
        convolution3 = F.relu(self.en3(convolution2))
        z = self.linear(convolution3)

        return z

class Decoder(nn.Module):
    def __init__(self, device, n_features, z_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_dim=z_dim
        self.device = device
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float64, device=device)
        self.de2 = nn.Linear(50, 100, dtype=torch.float64, device=device)
        self.de3 = nn.Linear(100, 200, dtype=torch.float64, device=device)
        self.de4 = nn.Linear(
            200, n_features, dtype=torch.float64, device=device)

    def forward(self, z):

        h4 = F.relu(self.de1(z))
        h5 = F.relu(self.de2(h4))
        h6 = F.relu(self.de3(h5))
        out = self.de4(h6)
        predicted = torch.tanh(out)
        return predicted

class Discriminator(nn.Module):
    def __init__(self, device, n_features, z_dim, dim_h=3,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_dim=z_dim
        self.device = device
        
        self.dim_z = z_dim
        self.dim_h = dim_h
        self.network = []
        self.network.extend([
            nn.Linear(self.dim_z, self.dim_h,dtype=torch.float64, device=device),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_h,dtype=torch.float64, device=device),
            nn.ReLU(),
            nn.Linear(self.dim_h,1,dtype=torch.float64, device=device),
            nn.Sigmoid(),
        ])
        self.network = nn.Sequential(*self.network)

    def forward(self, z):
        disc = self.network(z)

        return disc

