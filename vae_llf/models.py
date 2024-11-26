"""Implementation of the models for the Variational Autoencoder for Latent Feature Analysis"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import Callable

class VAEEncoder(nn.Module):
    """Empty class for generic VAE Encoder. 
    
    Args:
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.

    Methods
    -------
    reparameterize(mu, logvar):
        Performs the reparameterization trick

    divergence_loss(mu, logvar, reduction='sum'):
        Computes the Kullback-Leibler divergence between the latent distribution (mu,logvar) and N(0,1)
    """

    def __init__(self, latent_dim:int=20, kws_loss:dict={'reduction':'sum', 'β': 1.0}) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.kws_loss = kws_loss
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x:Tensor) -> Tensor:
        return x
    
    def predict_fn(self, X):
        self.eval()
        device = self.dummy_param.device
        with torch.no_grad():
            data = torch.from_numpy(X.astype("float32")).to(device)
            z, z_mean, z_logvar = self.forward(data)
        zvar = z_mean.cpu().detach().numpy()
        return zvar

    def reparameterize(self, mu:Tensor, logvar:Tensor) -> Tensor:
        """Performs the reparameterization trick

        Args:
            mu (Tensor): tensor of means
            logvar (Tensor): tensor of log(variance) 

        Returns:
            Tensor: sampled tensor at the Z layer
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def _kl_divergence(self, mu:Tensor, logvar:Tensor) -> Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1) # correct log(std) = log(var)/2

    
    def divergence_loss(self, z_params:dict=None, z_sampled:Tensor=None) -> Tensor:
        """Computes the Kullback-Leibler divergence between the latent distribution (mu,logvar) and N(0,1)

        Args:
            mu (Tensor): tensor of means
            logvar (Tensor): tensor of log(variance) 
            z_sampled: unused. 

        Returns:
            Tensor: KL divergence
        """
        β = self.kws_loss['β']
        kl_batch = self._kl_divergence(z_params['mu'], z_params['logvar'])
        if self.kws_loss['reduction']=='sum':
            return β*kl_batch.sum()
        else:
            return β*kl_batch.mean()

class MMDEncoder(VAEEncoder):

    def rbf_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""

        C = 2.0 * self.latent_dim*self.kws_loss['kernel_bandwidth']**2

        k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)

        return k
    
    def imq_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""

        Cbase = (
            2.0 * self.latent_dim*self.kws_loss['kernel_bandwidth']**2
        )

        k = 0

        for scale in self.kws_loss['scales']:
            C = scale * Cbase
            k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)

        return k
    
    def compute_mmd(self, z:Tensor, z_prior:Tensor) -> Tensor:
        """Calcula el MMD between x and y. 

        Args:
            x (Tensor): _description_
            y (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        N = z.shape[0]
        if self.kws_loss['kernel_choice'] == "rbf":
            k_z = self.rbf_kernel(z, z)
            k_z_prior = self.rbf_kernel(z_prior, z_prior)
            k_cross = self.rbf_kernel(z, z_prior)

        else:
            k_z = self.imq_kernel(z, z)
            k_z_prior = self.imq_kernel(z_prior, z_prior)
            k_cross = self.imq_kernel(z, z_prior)

        mmd_z = (k_z - k_z.diag().diag()).sum() / ((N - 1) * N)
        mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((N - 1) * N)
        mmd_cross = k_cross.sum() / (N**2)

        mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross
        return mmd_loss

    def divergence_loss(self, z_params:dict=None, z_sampled:Tensor=None) -> Tensor:
        """Computes the Maximum Mean Discrepancy (arXiv:0805.2368) between the sampled latent distribution (mu=z, logvar=None) and N(0,1)

        Args:
            mu (Tensor): tensor of means
            logvar (Tensor): tensor of log(variance) 

        Returns:
            Tensor: KL divergence
        """
        z_prior = torch.randn_like(z_sampled, device=z_sampled.device)
        kl_batch = self._kl_divergence(z_params['mu'], z_params['logvar'])
        mdd_batch = self.compute_mmd(z_sampled, z_prior)
        α = self.kws_loss['α']
        λ = self.kws_loss['λ']

        if self.kws_loss['reduction']=='sum':
            return (1-α)*kl_batch.sum(dim=0) + (α+λ-1)*mdd_batch 
        else:
            return (1-α)*kl_batch.mean(dim=0) + (α+λ-1)*mdd_batch 
        


class VAEDecoder(nn.Module):
    """Empty class for generic VAE Decoder. 
    
    Args:
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    
    Methods
    -------
    recon_loss(predictions, targets, reduction='sum'):
        Computes the VAE reconstruction loss assuming gaussian distribution of the data.
    """

    def __init__(self, latent_dim:int=20, recon_function:callable=F.mse_loss, kws_loss:dict={'reduction':'sum'}) -> None: 
        super().__init__()
        self.latent_dim = latent_dim
        self.recon_function = recon_function
        self.kws_loss = kws_loss

    def forward(self, x:Tensor) -> Tensor:
        return x
    
    def recon_loss(self, targets:Tensor, predictions:Tensor, mask:Tensor=None) -> Tensor:
        """Computes the VAE reconstruction loss assuming gaussian distribution of the data.

        Args:
            targets (Tensor): Target (original) values of the input data
            predictions (Tensor): Predicted values for the input sample
            reduction (str, optional): Aggregation function for the reconstruction loss. Defaults to 'sum'.

        Returns:
            Tensor: Reconstruction loss
        """
        # From arXiv calibrated decoder: arXiv:2006.13202v3
        # D is the dimensionality of x. 
        if mask is not None:
            targets = targets[:,mask]
            predictions = predictions[:,mask]
        r_loss = self.recon_function(predictions.reshape(targets.shape[0],-1), 
                                     targets.reshape(targets.shape[0],-1), 
                                     reduction='none').sum(dim=-1)
        # torch.pow(predictions-targets, 2).mean(dim=(1,2,3,4)) #+ D * self.logsigma
        if self.kws_loss['reduction']=='sum':
            return r_loss.sum(dim=0)
        else:
            return r_loss.mean(dim=0)

    
class DenseEncoder(VAEEncoder):
    """Implementation of 2-layer perceptron that works as an encoder for tabular data. 

    Args:
        input_dim (int): Dimension of the input tabular data
        intermediate_dim (int): Number of neurons in the hidden layer
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    """
    def __init__(self, input_dim:int, intermediate_dim:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2_mean = nn.Linear(intermediate_dim, self.latent_dim)
        self.fc2_logvar = nn.Linear(intermediate_dim, self.latent_dim)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = F.elu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar
    
class DenseMMDEncoder(MMDEncoder):
    """Implementation of 2-layer perceptron MMD that works as an encoder for tabular data. 

    Args:
        input_dim (int): Dimension of the input tabular data
        intermediate_dim (int): Number of neurons in the hidden layer
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    """
    def __init__(self, input_dim:int, intermediate_dim:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2_mean = nn.Linear(intermediate_dim, self.latent_dim)
        self.fc2_logvar = nn.Linear(intermediate_dim, self.latent_dim)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = F.elu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar

class DenseDecoder(VAEDecoder): 
    """Implementation of 2-layer perceptron that works as an decoder for tabular data. 

    Args:
        input_dim (int): Dimension of the input tabular data
        intermediate_dim (int): Number of neurons in the hidden layer
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    """
    def __init__(self, intermediate_dim:int, output_dim:int, *args, normalize_output:bool=True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.normalize_output = normalize_output
        self.fc1 = nn.Linear(self.latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, x:Tensor) -> Tensor: 
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        if self.normalize_output:
            x = F.sigmoid(x)
        return x


class GenVAE(nn.Module):
    """Generic implementation for a VAE, involving a encoder and decoder, and necessary fuctions. 

    Args:
        encoder (VAEEncoder): Encoder architecture to be used. 
        decoder (VAEDecoder): Decoder architecture to be used. 
    """
    def __init__(self, encoder:VAEEncoder, decoder:VAEDecoder):
        super().__init__()
        assert encoder.latent_dim == decoder.latent_dim, "latent_dim of encoder and decoder must be equal"

        self.encode = encoder
        self.decode = decoder

    def forward(self, x:Tensor) ->  tuple[Tensor, Tensor, Tensor, Tensor]:
        # Encode
        z, z_mean, z_logvar = self.encode(x)

        # Decode
        x_recon = self.decode(z)

        return z, z_mean, z_logvar, x_recon
    
    def loss_function(
            self, 
            x:Tensor, 
            x_recon:Tensor, 
            z_params:dict, 
            z_sampled:Tensor=None, 
            mask:Tensor=None,
            )  -> tuple[Tensor, Tensor, Tensor]:
        """ Generic loss function. 

        Args:
            x (Tensor): _description_
            x_recon (Tensor): _description_
            z_mean (Tensor): _description_
            z_logvar (Tensor): _description_
            beta (float, optional): _description_. Defaults to 1..
            reduction (str, optional): _description_. Defaults to 'sum'.

        Returns:
            tuple[Tensor, Tensor, Tensor]: _description_
        """
        divergence_loss = self.encode.divergence_loss(z_params, z_sampled)
        recon_loss = self.decode.recon_loss(x, x_recon, mask=mask)
        return recon_loss + divergence_loss, recon_loss, divergence_loss