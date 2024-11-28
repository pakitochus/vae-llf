# Overview
This package implements a Variational Autoencoder (VAE) for Latent Feature Analysis, designed to handle various datasets, including ADNI and DIAN. The VAE architecture consists of an encoder and decoder, allowing for efficient representation learning and reconstruction of input data.

# Easy Usage
To get started with this package, follow these steps:

[!ALERT]
This documentation is auto-generated from the docstrings in the code. The package is under active development, and we do not provide yet a stable API.

1. **Install Dependencies**: Ensure you have the required libraries installed. You can do this using pip:
   ```
   pip install -r requirements.txt
   ```

2. **Configuration**: Modify the `config/config_sample.yaml` file to specify your dataset parameters, model parameters, and training parameters. Make sure to set the correct paths for your datasets.

3. **Run Training**: Use the `train.py` script to start training your model. You can specify the configuration file as a command-line argument:
   ```
   python train.py config/config_sample.yaml
   ```

4. **Results**: After training, the results will be saved in a CSV file named `training_results.csv`, which contains the loss values for each epoch.

5. **Model Evaluation**: Utilize the provided utility functions in `vae_llf/utils.py` to evaluate your trained model on validation datasets.

For more detailed information, refer to the documentation within the code and the comments provided in each module.

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Acknowledgments
We acknowledge the contributions of the research community and the datasets used in this project.
