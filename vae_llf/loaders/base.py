import os
from typing import Callable, List, Optional, Tuple, Dict, Union
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
from .configuration import Config

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
