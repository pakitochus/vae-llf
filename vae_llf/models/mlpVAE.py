import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from .baseVAE import VAEEncoder, VAEDecoder
from .infoVAE import MMDEncoder
    
class DenseEncoder(VAEEncoder):
    """Implementation of a 2-layer perceptron that works as an encoder for tabular data. 

    Args:
        input_dim (int): Dimension of the input tabular data.
        intermediate_dim (int): Number of neurons in the hidden layer.
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    """
    def __init__(self, input_dim:int, intermediate_dim:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2_mean = nn.Linear(intermediate_dim, self.latent_dim)
        self.fc2_logvar = nn.Linear(intermediate_dim, self.latent_dim)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Sampled latent variable, mean, and log variance.
        """
        x = F.elu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar
    
class DenseMMDEncoder(MMDEncoder):
    """Implementation of a 2-layer perceptron MMD that works as an encoder for tabular data. 

    Args:
        input_dim (int): Dimension of the input tabular data.
        intermediate_dim (int): Number of neurons in the hidden layer.
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    """
    def __init__(self, input_dim:int, intermediate_dim:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2_mean = nn.Linear(intermediate_dim, self.latent_dim)
        self.fc2_logvar = nn.Linear(intermediate_dim, self.latent_dim)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the MMD encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Sampled latent variable, mean, and log variance.
        """
        x = F.elu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar

class DenseDecoder(VAEDecoder): 
    """Implementation of a 2-layer perceptron that works as a decoder for tabular data. 

    Args:
        intermediate_dim (int): Number of neurons in the hidden layer.
        output_dim (int): Dimension of the output data.
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
        normalize_output (bool, optional): Whether to normalize the output. Defaults to True.
    """
    def __init__(self, intermediate_dim:int, output_dim:int, *args, normalize_output:bool=True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.normalize_output = normalize_output
        self.fc1 = nn.Linear(self.latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, x:Tensor) -> Tensor: 
        """Forward pass through the decoder.

        Args:
            x (Tensor): Input latent variable.

        Returns:
            Tensor: Reconstructed output.
        """
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        if self.normalize_output:
            x = F.sigmoid(x)
        return x

