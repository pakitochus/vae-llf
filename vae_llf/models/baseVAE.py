import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

class VAEEncoder(nn.Module):
    """Base class for generic VAE Encoder. 
    
    Args:
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
        kws_loss (dict, optional): Keyword arguments for loss computation. Defaults to {'reduction':'sum', 'β': 1.0}.

    Methods
    -------
    reparameterize(mu, logvar):
        Performs the reparameterization trick.

    divergence_loss(z_params:dict=None, z_sampled:Tensor=None) -> Tensor:
        Computes the Kullback-Leibler divergence between the latent distribution (mu, logvar) and N(0,1).
    """

    def __init__(self, latent_dim:int=20, kws_loss:dict={'reduction':'sum', 'β': 1.0}) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.kws_loss = kws_loss
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x:Tensor) -> Tensor:
        return x
    
    def predict_fn(self, X):
        """Predicts the latent variables for the input data.

        Args:
            X (numpy.ndarray): Input data to predict latent variables.

        Returns:
            numpy.ndarray: Predicted latent variables.
        """
        self.eval()
        device = self.dummy_param.device
        with torch.no_grad():
            data = torch.from_numpy(X.astype("float32")).to(device)
            z, z_mean, z_logvar = self.forward(data)
        zvar = z_mean.cpu().detach().numpy()
        return zvar

    def reparameterize(self, mu:Tensor, logvar:Tensor) -> Tensor:
        """Performs the reparameterization trick.

        Args:
            mu (Tensor): Tensor of means.
            logvar (Tensor): Tensor of log(variance). 

        Returns:
            Tensor: Sampled tensor at the Z layer.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def _kl_divergence(self, mu:Tensor, logvar:Tensor) -> Tensor:
        """Computes the Kullback-Leibler divergence.

        Args:
            mu (Tensor): Tensor of means.
            logvar (Tensor): Tensor of log(variance).

        Returns:
            Tensor: KL divergence value.
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    
    def divergence_loss(self, z_params:dict=None, z_sampled:Tensor=None) -> Tensor:
        """Computes the Kullback-Leibler divergence between the latent distribution (mu, logvar) and N(0,1).

        Args:
            z_params (dict): Dictionary containing 'mu' and 'logvar'.
            z_sampled (Tensor, optional): Unused.

        Returns:
            Tensor: KL divergence.
        """
        β = self.kws_loss['β']
        kl_batch = self._kl_divergence(z_params['mu'], z_params['logvar'])
        if self.kws_loss['reduction']=='sum':
            return β * kl_batch.sum()
        else:
            return β * kl_batch.mean()
        



class VAEDecoder(nn.Module):
    """Base class for generic VAE Decoder. 
    
    Args:
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
        recon_function (callable, optional): Function to compute reconstruction loss. Defaults to F.mse_loss.
        kws_loss (dict, optional): Keyword arguments for loss computation. Defaults to {'reduction':'sum'}.

    Methods
    -------
    recon_loss(targets, predictions, mask=None) -> Tensor:
        Computes the VAE reconstruction loss assuming Gaussian distribution of the data.
    """

    def __init__(self, latent_dim:int=20, recon_function:callable=F.mse_loss, kws_loss:dict={'reduction':'sum'}) -> None: 
        super().__init__()
        self.latent_dim = latent_dim
        self.recon_function = recon_function
        self.kws_loss = kws_loss

    def forward(self, x:Tensor) -> Tensor:
        return x
    
    def recon_loss(self, targets:Tensor, predictions:Tensor, mask:Tensor=None) -> Tensor:
        """Computes the VAE reconstruction loss assuming Gaussian distribution of the data.

        Args:
            targets (Tensor): Target (original) values of the input data.
            predictions (Tensor): Predicted values for the input sample.
            mask (Tensor, optional): Mask to apply to the targets and predictions. Defaults to None.

        Returns:
            Tensor: Reconstruction loss.
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
        


class GenVAE(nn.Module):
    """Generic implementation for a VAE, involving an encoder and decoder, and necessary functions. 

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
        """Forward pass through the VAE.

        Args:
            x (Tensor): Input data.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: Sampled latent variable, mean, log variance, and reconstructed output.
        """
        z, z_mean, z_logvar = self.encode(x)
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
        """Generic loss function for the VAE.

        Args:
            x (Tensor): Original input data.
            x_recon (Tensor): Reconstructed output.
            z_params (dict): Dictionary containing 'mu' and 'logvar'.
            z_sampled (Tensor, optional): Sampled latent variables. Defaults to None.
            mask (Tensor, optional): Mask to apply to the input data. Defaults to None.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Total loss, reconstruction loss, and divergence loss.
        """
        divergence_loss = self.encode.divergence_loss(z_params, z_sampled)
        recon_loss = self.decode.recon_loss(x, x_recon, mask=mask)
        return recon_loss + divergence_loss, recon_loss, divergence_loss