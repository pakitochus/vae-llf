from typing import Callable, Optional
import pandas as pd
from .configuration import Config
from .base import TableDataset

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