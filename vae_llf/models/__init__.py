"""
Models (:mod:`vae_llf.models`)
==============================

.. currentmodule:: vae_llf.models

Implementation of the models for the Variational Autoencoder for Latent Feature Analysis.

This module contains the implementation of various encoder and decoder architectures 
for a Variational Autoencoder (VAE), including the generic VAE encoder and decoder, 
as well as specific implementations like DenseEncoder, DenseDecoder, MMDEncoder, 
and their respective loss functions.


Classes
-------

.. autosummary::
   :toctree: generated/

   VAEEncoder
   VAEDecoder
   GenVAE
   MMDEncoder
   DenseEncoder
   DenseMMDEncoder
   DenseDecoder

"""

from .baseVAE import VAEEncoder, VAEDecoder, GenVAE
from .infoVAE import MMDEncoder
from .mlpVAE import DenseEncoder, DenseMMDEncoder, DenseDecoder

__all__ = ['VAEEncoder', 'MMDEncoder', 'VAEDecoder', 'GenVAE', 'DenseEncoder', 'DenseMMDEncoder', 'DenseDecoder']
