"""
Utility functions for training and evaluating VAE models, as well as dataset management.

This module provides core functionality for:
    - Dataset management and creation
    - Model training and evaluation
    - Weight initialization
    - Training loop with early stopping

Functions:
    create_dataset_dictionary(config: Config) -> Dict[str, Dataset]:
        Creates a dictionary of datasets based on the configuration.

    epoch_train(model, normalizer, loader, optimizer, config, e) -> Tuple[torch.Tensor, Dict[str, float]]:
        Performs a single epoch of training on the given model.

    evaluate(model, normalizer, loader, config, e, return_outputs=False) -> Union[torch.Tensor, Tuple]:
        Evaluates the model on a given dataset.

    weights_init(m):
        Initializes the weights of linear layers using Xavier uniform initialization.

    train(model, optimizer, config, normalizer, train_loader, val_loader, ...) -> Tuple:
        Trains the model for multiple epochs, including validation and early stopping.

Example:
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
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tensorboardX import SummaryWriter

from loaders import ADNIDataset, DallasDataset, DIANDataset, NACCDataset, OASISDataset, Config

def create_dataset_dictionary(config: Config) -> Dict[str, Dataset]:
    """
    Creates a dictionary of datasets based on the provided configuration.

    Args:
        config (Config): Configuration object containing dataset specifications.

    Returns:
        Tuple[Dict[str, Dataset], List[str]]:
            - A dictionary where keys are dataset names and values are the corresponding Dataset objects.
            - A list of unique IDs that are present in all specified datasets.
    """
    datasets = {}
    if 'dallas' in config.datasets.keys():
        datasets['dallas'] = DallasDataset(config.datasets['dallas'], config)

    if 'nacc' in config.datasets.keys():
        datasets['nacc'] = NACCDataset(config.datasets['nacc'], config)

    if 'adni' in config.datasets.keys():
        datasets['adni'] = ADNIDataset(config.datasets['adni'], config)

    if 'dian' in config.datasets.keys():
        datasets['dian'] = DIANDataset(config.datasets['dian'], config)

    if 'oasis' in config.datasets.keys():
        datasets['oasis'] = OASISDataset(config.datasets['oasis'], config)

    id_list = list(set.intersection(*[set(datasets[el].features['id']) for el in config.datasets.keys()]))

    return datasets, id_list


def init_experiment(config):
    """Initialize the experiment folder.

    Args:
        config (Config): Configuration object.
    """
    if os.path.exists(config.model_name):
        shutil.rmtree(config.model_name)

    for subfolder in ['models', 'runs', 'results', 'figures']:
        os.makedirs(os.path.join(config.model_name, subfolder))

    shutil.copyfile('configs/config_ndim.yaml',
                    os.path.join(config.model_name, 'config.yaml'))


def epoch_train(model: torch.nn.Module, 
                normalizer: Callable[[torch.Tensor], torch.Tensor], 
                loader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                config: Config, 
                e: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Performs a single epoch of training on the given model.

    Args:
        model: The model to train.
        normalizer: Function to normalize input data.
        loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        config: Configuration object containing training parameters.
        e: Current epoch number.

    Returns:
        tuple: A tuple containing:
            - train_epoch (torch.Tensor): Tensor of losses for each batch.
            - dict_loss (dict): Dictionary of different loss components.
    """
    model.train()
    train_epoch = torch.empty((len(loader), 3)) # tensor to store losses
    with tqdm(loader) as t:
        for ix, batch in enumerate(t):
            t.set_description(f'E: {e}')
            optimizer.zero_grad()
            batch['data'] = normalizer(batch['data']).to(config.device)
            z, z_mean, z_logvar, x_recon = model(batch['data'])
            z_params = {'mu': z_mean, 'logvar': z_logvar}
            total_loss, recon_loss, divergence_loss = model.loss_function(batch['data'], x_recon, z_params, z, mask=None)
            dict_loss = dict(zip(['total', 'recon', 'div'], [total_loss, recon_loss, divergence_loss]))
            loss = total_loss
            t.set_postfix(loss=f"{dict_loss['total']:.2f}", 
                          rl=f"{dict_loss['recon']:.2f}", 
                          dl=f"{dict_loss['div']:.2f}",)
            loss.backward()
            optimizer.step()
            train_epoch[ix] = torch.Tensor(list(dict_loss.values())).detach().cpu()

    return train_epoch, dict_loss

def evaluate(model: torch.nn.Module, 
             normalizer: Callable[[torch.Tensor], torch.Tensor], 
             loader: DataLoader, 
             config: Config, 
             e: int, 
             return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]]]:
    """
    Evaluates the model on a given dataset.

    Args:
        model: The model to evaluate.
        normalizer: Function to normalize input data.
        loader: DataLoader for evaluation data.
        config: Configuration object containing evaluation parameters.
        e: Current epoch number.
        return_outputs: Whether to return model outputs.

    Returns:
        torch.Tensor or tuple: If return_outputs is False, returns mean validation losses.
                               If True, returns a tuple of (mean validation losses, model outputs).
    """
    model.eval()
    val_epoch = torch.empty((len(loader), 3)) # tensor to store losses

    if return_outputs:
        outputs = dict(td_recon=torch.Tensor(),
                       z_td = torch.Tensor(),
                       z_mu = torch.Tensor(),
                       z_lv = torch.Tensor(),
                       id = [],
                       visit = [],
                       input_data = torch.Tensor(),
                       dict_loss = {})
    with torch.no_grad():
        with tqdm(loader) as t:
            for ix, batch in enumerate(t):
                t.set_description(f'Eval: {e}')
                batch['data'] = normalizer(batch['data']).to(config.device)
                if return_outputs:
                    outputs['input_data'] = torch.cat((outputs['input_data'], batch['data'].cpu()))
                    outputs['id'] = outputs['id']+batch['id']
                    outputs['visit'] = outputs['visit']+batch['visit']
                z, z_mean, z_logvar, x_recon = model(batch['data'])
                z_params = {'mu': z_mean, 'logvar': z_logvar}
                loss, recon_loss, divergence_loss = model.loss_function(batch['data'], x_recon, z_params, z, mask=None)
                dict_loss = dict(zip(['total', 'recon', 'div'], [loss, recon_loss, divergence_loss]))
                if return_outputs:
                    outputs['td_recon'] = torch.cat((outputs['td_recon'], x_recon.detach().cpu()))
                    outputs['z_td'] = torch.cat((outputs['z_td'], z.detach().cpu()))
                    outputs['z_mu'] = torch.cat((outputs['z_mu'], z_mean.detach().cpu()))
                    outputs['z_lv'] = torch.cat((outputs['z_lv'], z_logvar.detach().cpu()))
                    outputs['dict_loss'] = dict_loss
                t.set_postfix(loss=f"{loss.item():.2f}")

                val_epoch[ix] = torch.Tensor(list(dict_loss.values())).detach().cpu()
    val_epoch = val_epoch.mean(axis=0)
    print(f'E {e}: Validation loss: {val_epoch[0]}')
        
    if return_outputs:
        return val_epoch, outputs
    else:
        return val_epoch
    
def weights_init(m: nn.Module) -> None:
    """
    Initializes the weights of a given neural network layer.

    This function applies the Xavier uniform initialization to the weights of 
    linear layers in the model. It is typically used to improve the convergence 
    of the training process.

    Args:
        m (nn.Module): The layer of the neural network to initialize. 
                       This function specifically checks for instances of 
                       nn.Linear.

    Returns:
        None: This function modifies the weights in place and does not return 
              any value.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def train(model: torch.nn.Module, 
          optimizer: torch.optim.Optimizer, 
          config: Any, 
          normalizer: Callable[[torch.Tensor], torch.Tensor], 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          start_epoch: int = 0, 
          end_epoch: int = 100, 
          writer: Optional[SummaryWriter] = None, 
          best_val_loss: Optional[float] = None) -> Tuple[torch.nn.Module, SummaryWriter, int, float]:
    """
    Trains the model for multiple epochs, including validation and early stopping.

    Args:
        model: The model to train.
        optimizer: Optimizer for model parameters.
        config: Configuration object containing training parameters.
        normalizer: Function to normalize input data.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        start_epoch: Starting epoch number.
        end_epoch: Ending epoch number.
        writer: TensorBoard SummaryWriter object.
        best_val_loss: Best validation loss from previous training, if any.

    Returns:
        tuple: A tuple containing:
            - model: The trained model.
            - writer: The TensorBoard SummaryWriter object.
            - e: The final epoch number.
            - best_val_loss: The best validation loss achieved.
    """
    if best_val_loss is None:
        best_val_loss = torch.inf
    count_iters = 0
    if writer is None:
        writer = SummaryWriter(os.path.join(config.model_name, 'runs', config.filename))
    train_loss = []
    for e in range(start_epoch, end_epoch):
        train_epoch, dict_loss = epoch_train(model, normalizer, train_loader, optimizer, config, e)

        train_loss.append(train_epoch.mean(dim=0))
        writer.add_scalar('Training Loss/total', sum(train_loss[-1]), e)
        for ix, key in enumerate(dict_loss.keys()):
            writer.add_scalar(f'Training Loss/{key}', train_loss[-1][ix], e)

        if e > 10: # leave the model warm up. 
            val_loss = evaluate(model, normalizer, val_loader, config, e, return_outputs=False)
            total_val_loss = val_loss[0]

            for ix, key in enumerate(dict_loss.keys()):
                writer.add_scalar(f'Validation Loss/{key}', val_loss[ix], e)

            if config.early_stopping:
                if total_val_loss <= best_val_loss + 1e-3: # threshold to save model if there is no significant difference 
                    print(f'VAL [{total_val_loss}<{best_val_loss}]: Saving model...')
                    best_val_loss = total_val_loss
                    torch.save(model.state_dict(), os.path.join(config.model_name, 'models', config.filename+".pth"))
                    count_iters = 0
                count_iters += 1
                if count_iters > config.max_iters:
                    print(f'{config.max_iters} with no improvement in validation. Ending loop')
                    torch.save(model.state_dict(), os.path.join(config.model_name, 'models', config.filename+"_END.pth"))
                    break

    return model, writer, e, best_val_loss