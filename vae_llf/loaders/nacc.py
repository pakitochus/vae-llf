import os
from typing import Callable, Optional
import pandas as pd
from .configuration import Config
from .base import TableDataset

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