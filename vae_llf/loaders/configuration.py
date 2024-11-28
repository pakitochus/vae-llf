from typing import List, Optional, Dict
from dataclasses import dataclass, field
import datetime
import yaml

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
    
    def to_yaml(self, file_path: str):
        """
        Saves the configuration settings to a YAML file.

        Args:
            file_path (str): Path to the YAML configuration file.
        """
        with open(file_path, 'w') as file:
            yaml.dump(self.__dict__, file)

    
def load_config(file_path: str) -> Config:
    """
    Loads configuration settings from a specified YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        Config: An instance of the Config class with the loaded settings.
    """
    return Config.from_yaml(file_path)