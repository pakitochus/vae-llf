import os
from typing import Callable, Optional
import pandas as pd
from .configuration import Config
from .base import TableDataset

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