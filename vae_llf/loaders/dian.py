import os
from typing import Callable, Optional
import pandas as pd
from .configuration import Config
from .base import TableDataset

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