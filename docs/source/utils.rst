 Utils
=====

.. module:: vae_llf.utils

This module provides utility functions for training and evaluating VAE models, as well as dataset management.

Dataset Management
----------------

.. function:: create_dataset_dictionary(config: Config) -> Dict[str, Dataset]

   Creates a dictionary of datasets based on the configuration.

   :param config: Configuration object containing dataset specifications
   :type config: Config
   :returns: A tuple containing (datasets dictionary, list of common IDs across datasets)
   :rtype: Tuple[Dict[str, Dataset], List[str]]

Training Functions
----------------

.. function:: epoch_train(model, normalizer, loader, optimizer, config, e)

   Performs a single epoch of training on the given model.

   :param model: The model to train
   :type model: torch.nn.Module
   :param normalizer: Function to normalize input data
   :type normalizer: Callable[[torch.Tensor], torch.Tensor]
   :param loader: DataLoader for training data
   :type loader: DataLoader
   :param optimizer: Optimizer for model parameters
   :type optimizer: torch.optim.Optimizer
   :param config: Configuration object containing training parameters
   :type config: Config
   :param e: Current epoch number
   :type e: int
   :returns: Tuple of (training losses tensor, loss components dictionary)
   :rtype: Tuple[torch.Tensor, Dict[str, float]]

.. function:: evaluate(model, normalizer, loader, config, e, return_outputs=False)

   Evaluates the model on a given dataset.

   :param model: The model to evaluate
   :type model: torch.nn.Module
   :param normalizer: Function to normalize input data
   :type normalizer: Callable[[torch.Tensor], torch.Tensor]
   :param loader: DataLoader for evaluation data
   :type loader: DataLoader
   :param config: Configuration object containing evaluation parameters
   :type config: Config
   :param e: Current epoch number
   :type e: int
   :param return_outputs: Whether to return model outputs
   :type return_outputs: bool
   :returns: Mean validation losses or tuple of (validation losses, model outputs)
   :rtype: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]]]

.. function:: weights_init(m)

   Initializes the weights of linear layers using Xavier uniform initialization.

   :param m: Module to initialize
   :type m: torch.nn.Module

.. function:: train(model, optimizer, config, normalizer, train_loader, val_loader, start_epoch=0, end_epoch=100, writer=None, best_val_loss=None)

   Trains the model for multiple epochs, including validation and early stopping.

   :param model: The model to train
   :type model: torch.nn.Module
   :param optimizer: Optimizer for model parameters
   :type optimizer: torch.optim.Optimizer
   :param config: Configuration object containing training parameters
   :type config: Any
   :param normalizer: Function to normalize input data
   :type normalizer: Callable[[torch.Tensor], torch.Tensor]
   :param train_loader: DataLoader for training data
   :type train_loader: DataLoader
   :param val_loader: DataLoader for validation data
   :type val_loader: DataLoader
   :param start_epoch: Starting epoch number (default: 0)
   :type start_epoch: int
   :param end_epoch: Ending epoch number (default: 100)
   :type end_epoch: int
   :param writer: TensorBoard SummaryWriter object (default: None)
   :type writer: Optional[SummaryWriter]
   :param best_val_loss: Best validation loss from previous training (default: None)
   :type best_val_loss: Optional[float]
   :returns: Tuple of (trained model, writer, final epoch, best validation loss)
   :rtype: Tuple[torch.nn.Module, SummaryWriter, int, float]

Example Usage
------------

Here's a basic example of how to use the training functions:

.. code-block:: python

   import torch
   from vae_llf.utils import train, weights_init
   
   # Initialize model and apply weight initialization
   model = YourVAEModel()
   model.apply(weights_init)
   
   # Setup training components
   optimizer = torch.optim.Adam(model.parameters())
   config = YourConfig()
   normalizer = lambda x: x  # Your normalization function
   
   # Train the model
   model, writer, final_epoch, best_loss = train(
       model=model,
       optimizer=optimizer,
       config=config,
       normalizer=normalizer,
       train_loader=train_loader,
       val_loader=val_loader
   ) 