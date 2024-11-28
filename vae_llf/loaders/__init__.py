"""
Loaders (:mod:`vae_llf.loaders`)
================================

.. currentmodule:: vae_llf.loaders

This module contains classes and functions for loading and managing datasets for a machine learning model.
It includes configuration settings, dataset classes for various data sources, and methods for data preprocessing.


Classes
-------

.. autosummary::
   :toctree: generated/

   Config
   TableDataset
   DIANDataset
   ADNIDataset
   NACCDataset
   DallasDataset
   OASISDataset

Functions
---------

.. autosummary::
   :toctree: generated/

   load_config
"""

from .configuration import Config, load_config
from .base import TableDataset
from .dian import DIANDataset
from .adni import ADNIDataset
from .nacc import NACCDataset
from .dallas import DallasDataset
from .oasis import OASISDataset

__all__ = ['Config', 'load_config', 'TableDataset', 'DIANDataset', 'ADNIDataset', 'NACCDataset', 'DallasDataset', 'OASISDataset']
