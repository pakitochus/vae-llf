.. VAE - Longitudinal Latent Feature analysis documentation master file, created by
   sphinx-quickstart on Wed Nov 27 09:55:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VAE - Longitudinal Latent Feature analysis documentation
========================================================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

This package implements a Variational Autoencoder (VAE) for Latent Feature Analysis, designed 
to handle various datasets, including ADNI and DIAN. The VAE architecture consists of an encoder 
and decoder, allowing for efficient representation learning and reconstruction of input data.

.. note::

   This project is under active development.



Examples
--------

    >>> import torch
    >>> from vae_llf.utils import train, weights_init
    >>> 
    >>> # Initialize model and apply weight initialization
    >>> model = YourVAEModel()
    >>> model.apply(weights_init)
    >>> 
    >>> # Setup training components
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> config = YourConfig()
    >>> normalizer = lambda x: x  # Your normalization function
    >>> 
    >>> # Train the model
    >>> model, writer, final_epoch, best_loss = train(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     config=config,
    ...     normalizer=normalizer,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader
    ... )

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   vae_llf
   readme

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`