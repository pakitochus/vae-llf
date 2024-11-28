import torch
from torch import Tensor
from .baseVAE import VAEEncoder

class MMDEncoder(VAEEncoder):
    """Extension of VAEEncoder that computes Maximum Mean Discrepancy (MMD).

    Methods
    -------
    rbf_kernel(z1, z2):
        Computes the RBF kernel between two sets of latent variables.

    imq_kernel(z1, z2):
        Computes the Inverse Multiquadric kernel between two sets of latent variables.

    compute_mmd(z:Tensor, z_prior:Tensor) -> Tensor:
        Computes the Maximum Mean Discrepancy between two distributions.

    divergence_loss(z_params:dict=None, z_sampled:Tensor=None) -> Tensor:
        Computes the divergence loss using MMD.
    """

    def rbf_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise RBF kernel computation.

        Args:
            z1 (Tensor): First set of latent variables.
            z2 (Tensor): Second set of latent variables.

        Returns:
            Tensor: RBF kernel matrix.
        """
        C = 2.0 * self.latent_dim * self.kws_loss['kernel_bandwidth']**2
        k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)
        return k
    
    def imq_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise Inverse Multiquadric kernel computation.

        Args:
            z1 (Tensor): First set of latent variables.
            z2 (Tensor): Second set of latent variables.

        Returns:
            Tensor: Inverse Multiquadric kernel matrix.
        """
        Cbase = 2.0 * self.latent_dim * self.kws_loss['kernel_bandwidth']**2
        k = 0

        for scale in self.kws_loss['scales']:
            C = scale * Cbase
            k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)

        return k
    
    def compute_mmd(self, z:Tensor, z_prior:Tensor) -> Tensor:
        """Calculates the Maximum Mean Discrepancy (MMD) between two distributions.

        Args:
            z (Tensor): Sampled latent variables.
            z_prior (Tensor): Prior latent variables.

        Returns:
            Tensor: MMD loss value.
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
        """Computes the divergence loss using Maximum Mean Discrepancy (MMD).

        Args:
            z_params (dict): Dictionary containing 'mu' and 'logvar'.
            z_sampled (Tensor): Sampled latent variables.

        Returns:
            Tensor: Divergence loss value.
        """
        z_prior = torch.randn_like(z_sampled, device=z_sampled.device)
        kl_batch = self._kl_divergence(z_params['mu'], z_params['logvar'])
        mdd_batch = self.compute_mmd(z_sampled, z_prior)
        α = self.kws_loss['α']
        λ = self.kws_loss['λ']

        if self.kws_loss['reduction']=='sum':
            return (1-α) * kl_batch.sum(dim=0) + (α + λ - 1) * mdd_batch 
        else:
            return (1-α) * kl_batch.mean(dim=0) + (α + λ - 1) * mdd_batch 
        