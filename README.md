# Overview
This package implements a Variational Autoencoder (VAE) for Latent Feature Analysis, designed initially for
tabular datasets. The VAE architecture consists of an encoder and decoder, allowing for efficient representation learning and reconstruction of input data. It allows easy customization of the model architecture, such as the number of layers and nodes per layer, and confiiguration via a YAML file.

> [!CAUTION]
> This documentation is auto-generated from the docstrings in the code. The package is under active development, and we do not provide yet a stable API.

# Easy Usage
To get started with this package, follow these steps:

1. **Install Dependencies**: Ensure you have the required libraries installed. You can do this using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**: Modify the `config/config_sample.yaml` file to specify your dataset parameters, model parameters, and training parameters. Make sure to set the correct paths for your datasets.

3. **Run Training**: Use the `train.py` script to start training your model. You can specify the configuration file as a command-line argument:
   ```bash
   python train.py config/config_sample.yaml
   ```

4. **Results**: After training, the results will be saved in a CSV file named `training_results.csv`, which contains the loss values for each epoch.

5. **Model Evaluation**: Utilize the provided utility functions in `vae_llf/utils.py` to evaluate your trained model on validation datasets.

For more detailed information, refer to the documentation within the code and the comments provided in each module.

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Acknowledgments
This software is part of the [LATiDOS](https://pakitochus.github.io/research/2023-09-01-project-latidos/) project (ref. PID2022-137629OA-I00) funded by MICIU/AEI/10.13039/501100011033 and by ERDF/EU. 
