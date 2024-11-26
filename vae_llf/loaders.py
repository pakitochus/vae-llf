"""
This module contains classes and functions for loading and managing datasets for a machine learning model.
It includes configuration settings, dataset classes for various data sources, and methods for data preprocessing.

Classes:
    - Config: A dataclass for managing configuration settings for the datasets and model.
    - TableDataset: A base class for creating PyTorch datasets from tabular data.
    - DIANDataset: A dataset class for loading and processing DIAN data.
    - ADNIDataset: A dataset class for loading and processing ADNI data.
    - NACCDataset: A dataset class for loading and processing NACC data.
    - DallasDataset: A dataset class for loading and processing Dallas data.
    - OASISDataset: A dataset class for loading and processing OASIS data.

Functions:
    - load_config: Loads configuration settings from a YAML file and returns a Config object.
"""

import os
from typing import AnyStr, Callable, List, Optional, Tuple, Dict, Union
from dataclasses import dataclass, field
import torch
from torch import Tensor
import torch.utils.data as Dataset
import numpy as np
import pandas as pd
import datetime
import yaml
import random

@dataclass
class Config:
    """
    Configuration settings for datasets and model training.

    Attributes:
        datasets (Dict[str, str]): Dictionary of dataset names and paths.
        batch_size (int): Size of each batch during training.
        modality (str): Modality of the data (e.g., 'mri').
        selection (List[str]): Features to select from the dataset.
        uptake_normalization (Optional[str]): Type of uptake normalization to apply.
        subject_norm (str): Method for subject normalization.
        only_id (bool): Flag to indicate if only IDs should be included.
        exclude_nan (bool): Flag to indicate if rows with NaN values should be excluded.
        exclude_ids (List[int]): List of IDs to exclude from the dataset.
        train_val_split (List[float]): Proportions for splitting the dataset into training and validation sets.
        ddata (int): Dimension of the data.
        interm_dim (int): Intermediate dimension for model architecture.
        kws_enc_loss (dict): Configuration for kernel-wise encoder loss.
        recon_function (str): Reconstruction function to use.
        kws_dec_loss (dict): Configuration for kernel-wise decoder loss.
        out_norm (bool): Flag to indicate if output normalization should be applied.
        model_name (str): Name of the model.
        div_loss (str): Type of divergence loss to use.
        beta (float): Weight for the loss function.
        lr (float): Learning rate for the optimizer.
        random_seed (int): Seed for random number generation.
        early_stopping (bool): Flag to indicate if early stopping should be used.
        n_epochs (int): Number of epochs for training.
        max_iters (int): Maximum iterations for training.
        savefigs (bool): Flag to indicate if figures should be saved.
        device (str): Device to use for training (e.g., 'cuda').
    """

    # Data config
    datasets: Dict[str, str] = field(default_factory=lambda: {'dian': 'path/to/dian'})
    batch_size: int = 128
    modality: str = 'mri'
    selection: List[str] = field(default_factory=lambda: ['V'])
    uptake_normalization: Optional[str] = None
    subject_norm: str = 'std'
    only_id: bool = True
    exclude_nan: bool = True
    exclude_ids: List[int] = field(default_factory=lambda: [])
    train_val_split: List[float] = field(default_factory=lambda: [0.8, 0.2])

    # Model config
    ddata: int = 3
    interm_dim: int = 256
    kws_enc_loss: dict = field(default_factory=lambda: {'kernel_choice': 'rbf', 'reduction': 'mean', 'α': 0, 'λ': 1e-3, 'kernel_bandwidth': 1})
    recon_function: str = 'mae'
    kws_dec_loss: dict = field(default_factory=lambda: {'reduction': 'mean'})
    out_norm: bool = False
    model_name: str = None

    # Loss config
    div_loss: str = 'MMD'
    beta: float = 1.0
    lr: float = 2e-3

    # Train config
    random_seed: int = 0
    early_stopping: bool = True
    n_epochs: int = 100000
    max_iters: int = 1000
    savefigs: bool = False
    device: str = 'cuda'

    def __post_init__(self):
        """Post-initialization processing to set up the configuration."""
        self._datetime_init = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        exp_name = self.generate_filename()
        self.filename = exp_name
        self.model_name = "_".join([self._datetime_init, self.model_name])
        
    def generate_filename(self) -> str:
        """
        Generates a unique filename based on the configuration settings.

        Returns:
            str: The generated filename.
        """
        exp_name = f'model[{self.div_loss};red:{self.kws_enc_loss["reduction"]}'
        if 'α' in self.kws_enc_loss.keys():
            exp_name += f";α:{self.kws_enc_loss['α']};λ:{self.kws_enc_loss['λ']}"
        exp_name += f']({self.interm_dim}-{self.ddata})]'
        exp_name += f'_D[{self.modality}({"-".join(self.selection)})_{self.uptake_normalization}_norm{self.subject_norm}_bs{self.batch_size}]'
        exp_name += f'_O[Adam_{self.lr}_e{self.n_epochs}]'
        exp_name += f'_{self._datetime_init}'
        return exp_name

    @classmethod
    def from_yaml(cls, file_path: str) -> 'Config':
        """
        Loads configuration settings from a YAML file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            Config: An instance of the Config class populated with settings from the file.
        """
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

    
def load_config(file_path: str) -> Config:
    """
    Loads configuration settings from a specified YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        Config: An instance of the Config class with the loaded settings.
    """
    return Config.from_yaml(file_path)



class TableDataset(Dataset):
    """
    A base class for creating PyTorch datasets from tabular data.

    This class is intended to be extended for specific datasets. It provides common methods for loading,
    processing, and accessing data.

    Attributes:
        DBDIR (str): Directory where the dataset files are located.
        config (Config): Configuration settings for the dataset.
        name (str): Name of the dataset.
        datafiles (Optional[dict]): Dictionary of data files associated with the dataset.
        transform (Optional[Callable]): Optional transform to be applied to the data.
    """

    def __init__(self, DBDIR: str, config: Config, name: str = None, datafiles: Optional[dict] = None, transform: Optional[Callable] = None):
        """
        Initializes the TableDataset with the given parameters.

        Args:
            DBDIR (str): Directory where the dataset files are located.
            config (Config): Configuration object containing dataset parameters.
            name (str, optional): Name of the dataset. Defaults to None.
            datafiles (Optional[dict], optional): Dictionary of data files. Defaults to None.
            transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.name = name
        self.datafiles = datafiles
        self._load_dataframe(DBDIR)
        self.exclude_nan = self.config.exclude_nan
        self.only_id = self.config.only_id
        self.norm = None if self.config.subject_norm=='None' else self.config.subject_norm
        self.transform = transform

        if self.only_id:
            self._set_id_dataframe(f'features/{self.name}/structures_{self.config.modality}_{self.name}.csv')
        self._update_variables()

        if self.config.exclude_ids not in [None, []]:
            self._exclude_ids()

        if self.exclude_nan: 
            self._exclude_nan()

    def _load_dataframe(self, DBDIR: str) -> pd.DataFrame:
        """
        Loads the dataset into a DataFrame.

        Args:
            DBDIR (str): Directory where the dataset files are located.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError
    
    def _update_variables(self):
        """
        Updates the internal variables of the dataset based on the current state of the dataframe.

        This method extracts the following information from the dataframe:
        - features: The columns of the dataframe, representing the features of the dataset.
        - patno: The patient numbers extracted from the dataframe index.
        - patnos: Unique patient numbers derived from patno.
        - visit: The visit numbers extracted from the dataframe index.
        - ref: A reference dataframe that is aligned with the current dataframe to maintain consistency.

        This method should be called whenever the dataframe is modified to ensure that the internal state
        reflects the current data.
        """
        #check whether self has "features" attribute
        if hasattr(self, 'features'):
            self.features = self.features.loc[self.dataframe.columns]
        else:
            self.features = self.dataframe.columns
        self.patno = self.dataframe.index.get_level_values('id').to_numpy().astype(str)
        self.patnos = np.unique(self.patno)
        self.visit = self.dataframe.index.get_level_values('visit').to_numpy().astype(str)
        self.ref = self.ref.loc[self.dataframe.index]  # keep consistency with the dataframe

    def patno_split_dataset(self, rates: List[float] = [0.8, 0.2], random_seed: int = 10) -> Tuple[List[int], ...]:
        """
        Splits the dataset into training, validation, and possibly test sets based on patient number (PATNO).

        Args:
            rates (list, optional): Proportions to split the dataset. Defaults to [0.8, 0.2].
            random_seed (int, optional): Seed for random number generator. Defaults to 10.

        Returns:
            Tuple[List[int]]: Indices for train and validation sets.
        """
        random.seed(random_seed)
        random.shuffle(self.patnos)
        N = len(self.patnos)
        subjects = (self.patnos[:round(rates[0]*N)],)
        if len(rates)==2:
            subjects += (self.patnos[round(rates[0]*N):],)
        elif len(rates)==3:
            subjects += (self.patnos[round(rates[0]*N):round(rates[0]*N)+round(rates[1]*N)],)
            subjects += (self.patnos[round(rates[0]*N)+round(rates[1]*N):],)

        patients = ()
        for sub_list in subjects:
            patients += ([ix for ix, el in enumerate(self.patno) if el in sub_list], )

        return patients
        
    def _getall_(self) -> dict:
        """
        Retrieves all data from the dataset.

        Returns:
            dict: A dictionary containing the data, patient IDs, visit numbers, and dataset name.
        """
        data = torch.from_numpy(self.dataframe.values.squeeze().astype('float32'))

        if self.norm is not None:
            if self.norm=='ref':
                ref = torch.from_numpy(self.ref.values.astype('float32'))
                data = self._normalize(data, ref=ref)
            else:
                data = self._normalize(data)

        return {'data': data, 'id': list(self.patno), 'visit': self.visit, 'name': self.name}
    
    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        """
        Returns the image and the target data for a given index.

        Args:
            index (int): Index

        Returns:
            dict: A dictionary containing the image and target data
        """
        data = torch.from_numpy(self.dataframe.iloc[index].values.squeeze().astype('float32'))

        if self.norm is not None:
            if self.norm=='ref':
                ref = torch.tensor(self.ref.iloc[index], dtype=torch.float32)
                data = self._normalize(data, ref=ref)
            else:
                data = self._normalize(data)

        return {'data': data, 'id': str(self.patno[index]), 'visit': self.visit[index], 'name': self.name}

    def __len__(self) -> int:
        """
        Returns the total number of images.

        Returns:
            int: Total number of images
        """
        return len(self.patno)
    
    def _normalize(self, X: Tensor, ref: Optional[float] = None) -> Tensor:
        """
        Normalizes the input tensor based on the specified normalization method.

        Args:
            X (Tensor): Input tensor to normalize.
            ref (Optional[float]): Reference value for normalization, if applicable.

        Returns:
            Tensor: Normalized tensor.
        """
        assert self.norm in ['mean', 'std', 'range', 'ref']
        norm = torch.nanmean(X, dim=0)
        if self.norm=='range':
            ref = torch.min(torch.nan_to_num(X, nan=1e4), dim=0).values
            max = torch.max(torch.nan_to_num(X, nan=-1e4), dim=0).values
            norm = (max-ref)
            norm[norm==0] = 1
        elif self.norm=='mean':
            norm[norm==0] = 1 # in mean norm norm is \mu. 
            ref = torch.zeros_like(norm) # and ref=0
        elif self.norm=='std':
            ref = norm.detach().clone() # in std \mu becomes ref. 
            norm = torch.std(torch.nan_to_num(X), dim=0)
            ref[ref==0] = 1
            norm[norm==0] = 1
        elif self.norm=='ref':
            norm = ref
            ref = torch.zeros_like(norm)
        Xnorm = (X-ref)/norm
        return Xnorm
    
    def _exclude_ids(self):
        """
        Excludes specified IDs from the dataset based on the configuration settings.
        """
        include_ids = [el for el in self.features.reset_index().set_index('id').index.values if el not in self.config.exclude_ids]
        self._filter_dataframe_from_id_list(include_ids)
        self._update_variables()

    def _exclude_nan(self):
        """
        Excludes rows with NaN values from the dataset.
        """
        self.dataframe = self.dataframe.dropna(axis=0)
        if self.norm=='ref':
            self.ref = self.ref.dropna(axis=0)
            self.dataframe = self.dataframe.loc[self.ref.index]
        self._update_variables()
    
    def _filter_dataframe_from_id_list(self, id_list):
        """
        Filters the dataframe based on a list of IDs.

        Args:
            id_list (list): List of IDs to include in the filtered dataframe.
        """
        proxy = self.features.reset_index().set_index('id')
        col_select = proxy.loc[id_list, 'index'].values
        self.dataframe = self.dataframe[col_select]
        self._update_variables()
    
    def _set_id_dataframe(self, features_path: str):
        """
        Sets the ID dataframe based on the features CSV file.

        Args:
            features_path (str): Path to the features CSV file.
        """
        features_df = pd.read_csv(os.path.join(features_path))
        features_id = features_df.loc[features_df['id'].isna()==False]
        order_list = list(features_id[['feature', 'id']].set_index('id').sort_index().to_records())
        col_intersect = list(set(features_id['feature']).intersection(set(self.dataframe.columns)))
        self.features = pd.DataFrame.from_records([tuple for tuple in order_list for x in col_intersect if tuple[1] == x], columns=['id', 'feature']).set_index('feature')
        self.dataframe = self.dataframe[self.features.index]


class DIANDataset(TableDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class loads data from the different csv files of DIAN and returns a dictionary containing the file data and associated target data.
    """

    def __init__(self, DBDIR: str, config: Config, transform: Optional[Callable] = None):
        """
        Initializes the DIAN dataset.

        Args:
            DBDIR (str): Path to the directory containing DIAN data files.
            config (Config): Configuration object containing dataset parameters including:
                - modality (str): Data modality ('MRI', 'PIB', or 'FDG')
                - selection (List[str]): Features to select (e.g., ['V'] for volume)
                - uptake_normalization (Optional[str]): Type of uptake normalization
                - subject_norm (str): Subject normalization method ('std', 'mean', 'range', or 'None')
                - only_id (bool): Whether to only include features with IDs
                - exclude_nan (bool): Whether to exclude rows with NaN values
            transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.

        The dataset loads data from CSV files and processes it according to the config:
        1. Loads raw data based on modality and selection criteria
        2. Optionally filters features to only those with IDs
        3. Optionally removes rows with NaN values
        4. Applies specified normalization
        """
        super().__init__(DBDIR, config, name='dian', 
                         datafiles={'MRI': 'IMAGING_D1727A.csv',
                                    'PIB': 'pib_D1727A.csv',
                                    'FDG': 'FDG_D1727A.csv'}, 
                         transform=transform)
        

    def _load_dataframe(self, DBDIR: str) -> pd.DataFrame:
        """
        Loads the DIAN dataset into a DataFrame.

        Args:
            DBDIR (str): Directory where the DIAN data files are located.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        # Selection: level 2 features.  
        dataframe = pd.read_csv(os.path.join(DBDIR, self.datafiles[self.config.modality.upper()]))
        dataframe = dataframe.set_index(['newid14', 'visit'])
        #rename dataframe index to 'id' and 'visit'
        dataframe.index = dataframe.index.set_names(['id', 'visit'])
        features = pd.read_csv(os.path.join(f'features/{self.name}/structures_{self.config.modality.lower()}_{self.name}.csv'))
        if self.config.uptake_normalization:
            col_select = pd.concat([features.loc[(features['3']==el)&(features['1']==self.config.uptake_normalization), 'feature'] for el in self.config.selection], axis=0)
        else:
            col_select = pd.concat([features.loc[features['3']==el, 'feature'] for el in self.config.selection], axis=0)
        self.dataframe = dataframe[col_select.values]
        self.ref = dataframe['MR_TOTV_INTRACRANIAL']/100

class ADNIDataset(TableDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class loads images from a csv file and returns a dictionary containing the image and associated target data.
    """
    def __init__(self, DBDIR: str, config: Config, transform: Optional[Callable] = None):
        """
        Initializes the ADNI dataset.

        Args:
            DBDIR (str): Path to the directory containing ADNI data files.
            config (Config): Configuration object containing dataset parameters.
            transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        """
        super().__init__(DBDIR=DBDIR, config=config, name='adni', 
                         datafiles={'MRI': 'data_merged_index_reset_unique.csv'}, 
                         transform=transform)

    def _load_dataframe(self, DBDIR: str) -> pd.DataFrame:
        """
        Loads the ADNI dataset into a DataFrame.

        Args:
            DBDIR (str): Directory where the ADNI data files are located.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        # Selection: level 2 features.  
        dataframe = pd.read_csv(DBDIR+'/data_merged_index_reset_unique.csv', index_col=[0,1])
        dataframe.index = dataframe.index.set_names(['newid14', 'visit'])
        #rename dataframe index to 'id' and 'visit'
        if dataframe.index.is_unique==False:
            dataframe = dataframe.loc[dataframe.index.duplicated()==False]
        dataframe.index = dataframe.index.set_names(['id', 'visit'])
        self.dataframe = dataframe[[el for el in dataframe.columns if el.startswith('ST')]]

        if self.config.selection[0]=='V': 
            self.dataframe = self.dataframe[[el for el in self.dataframe.columns if 'V' in el]]
        elif self.config.selection[0]=='T': 
            self.dataframe = self.dataframe[[el for el in self.dataframe.columns if 'TA' in el]]
        else:
            raise ValueError("Selection not recognised")

        self.ref = self.dataframe['ST10CV']/100


class NACCDataset(TableDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class loads images from a csv file and returns a dictionary containing the image and associated target data.
    """
    def __init__(self, DBDIR: str, config: Config, transform: Optional[Callable] = None):
        """
        Initializes the NACC dataset.

        Args:
            DBDIR (str): Path to the directory containing NACC data files.
            config (Config): Configuration object containing dataset parameters.
            transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        """
        super().__init__(DBDIR, config, name='nacc', 
                         datafiles={'MRI': 'mri_nacc.csv',
                            'PIB': 'amyloid_nacc.csv',
                                    'FDG': 'fdg_nacc.csv',
                                    'TAU': 'tau_nacc.csv'}, 
                         transform=transform)

    def _load_dataframe(self, DBDIR: str) -> pd.DataFrame:
        """
        Loads the NACC dataset into a DataFrame.

        Args:
            DBDIR (str): Directory where the NACC data files are located.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        # Selection: level 2 features.  
        dataframe = pd.read_csv(os.path.join(DBDIR, self.datafiles[self.config.modality.upper()]))
        dataframe = dataframe.set_index(['NACCID', 'NACCVNUM'])
        #rename dataframe index to 'id' and 'visit'
        dataframe.index = dataframe.index.set_names(['id', 'visit'])

        features = pd.read_csv(os.path.join(f'features/nacc/structures_{self.config.modality.lower()}_nacc.csv'))
        col_select = pd.concat([features.loc[features['type']==el, 'feature'] for el in self.config.selection], axis=0)

        self.dataframe = dataframe[col_select.values]
        if self.config.modality.lower()=='mri':
            self.ref = self.dataframe['CEREBRUMTCV']*10 #convert from cc to mm^3 ref*1000/100


class DallasDataset(TableDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class loads images from a csv file and returns a dictionary containing the image and associated target data.
    """
    def __init__(self, DBDIR: str, config: Config, transform: Optional[Callable] = None):
        """
        Initializes the Dallas dataset.

        Args:
            DBDIR (str): Path to the directory containing Dallas data files.
            config (Config): Configuration object containing dataset parameters.
            transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        """
        super().__init__(DBDIR, config, name='dallas', 
                         datafiles={'MRI': 'mri_dallas.csv',
                                    'PIB': 'amyloid_dallas.csv',
                                    'TAU': 'tau_dallas.csv'}, 
                         transform=transform)

    def _load_dataframe(self, DBDIR: str) -> pd.DataFrame:
        """
        Loads the Dallas dataset into a DataFrame.

        Args:
            DBDIR (str): Directory where the Dallas data files are located.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        # Selection: level 2 features.  
        dataframe = pd.read_csv(os.path.join(DBDIR, self.datafiles[self.config.modality.upper()]))
        dataframe = dataframe.set_index(['id', 'visit'])

        features = pd.read_csv(os.path.join(f'features/{self.name}/structures_{self.config.modality.lower()}_{self.name}.csv'))
        col_select = pd.concat([features.loc[features['type']==el, 'feature'] for el in self.config.selection], axis=0)

        self.dataframe = dataframe[col_select.values]
        if self.config.modality.lower()=='mri':
            self.ref = self.dataframe['Total_Intracranial_Vol']/100 



class OASISDataset(TableDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class loads images from a csv file and returns a dictionary containing the image and associated target data.
    """
    def __init__(self, DBDIR: str, config: Config, transform: Optional[Callable] = None):
        """
        Initializes the OASIS dataset.

        Args:
            DBDIR (str): Path to the directory containing OASIS data files.
            config (Config): Configuration object containing dataset parameters.
            transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        """
        super().__init__(DBDIR, config, name='oasis', 
                         datafiles={'MRI': 'mri_oasis.csv'}, 
                         transform=transform)

    def _load_dataframe(self, DBDIR: str) -> pd.DataFrame:
        """
        Loads the OASIS dataset into a DataFrame.

        Args:
            DBDIR (str): Directory where the OASIS data files are located.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        # Selection: level 2 features.  
        dataframe = pd.read_csv(os.path.join(DBDIR, self.datafiles[self.config.modality.upper()]))
        dataframe = dataframe.set_index(['id', 'visit'])

        features = pd.read_csv(os.path.join(f'features/{self.name}/structures_{self.config.modality.lower()}_{self.name}.csv'))
        col_select = pd.concat([features.loc[features['type']==el, 'feature'] for el in self.config.selection], axis=0)

        self.dataframe = dataframe[col_select.values]
        if self.config.modality.lower()=='mri':
            self.ref = self.dataframe['IntraCranial_Vol']/100 