import os
import sys
import pandas as pd
import torch
from vae_llf.loaders import load_config
from vae_llf.models import DenseEncoder, VAEDecoder, GenVAE  # Adjust based on your model choice
from vae_llf.utils import init_experiment, train, weights_init, create_combined_dataloaders, evaluate

def main(config_path: str):
    # Load configuration
    config = load_config(config_path)

    # init folder structure for experiment outputs
    init_experiment(config)

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # Initialize dataset and dataloader
    train_loader, val_loader, test_loader = create_combined_dataloaders(config)
    data_dim = len(train_loader.dataset.datasets[0].features)
    
    # Initialize model
    encoder = DenseEncoder(input_dim=data_dim, 
                           intermediate_dim=config.interm_dim, latent_dim=config.ddata)
    decoder = VAEDecoder(latent_dim=config.ddata)  # Adjust based on your model choice
    data_vae = GenVAE(encoder, decoder).to(device)

    normalize = torch.nn.BatchNorm1d(data_dim, momentum=None)

    data_vae = data_vae.to(config.device)
    data_vae.apply(weights_init)  # initialize weights.

    # Define optimizer
    optimizer = torch.optim.AdamW(data_vae.parameters(), lr=config.lr)

    # Training loop
    data_vae, writer, e, bvl = train(data_vae, optimizer, config, normalize, 
                                     train_loader, val_loader, start_epoch=0, 
                                     end_epoch=config.n_epochs)
    
    # Load and evaluate best performing model
    data_vae.load_state_dict(torch.load(os.path.join('runs',
        config.model_name, 'models', config.filename+".pth")))
    data_vae = data_vae.to(config.device)

    train_loss, outputs_train = evaluate(
        data_vae, normalize, train_loader, config, 0, return_outputs=True)
    val_loss, outputs_val = evaluate(
        data_vae, normalize, val_loader, config, 0, return_outputs=True)
    test_loss, outputs_test = evaluate(
        data_vae, normalize, test_loader, config, 0, return_outputs=True)

    losses = dict(
        train=outputs_train['dict_loss'],
        validation=outputs_val['dict_loss'],
        test=outputs_test['dict_loss']
    )

    data = pd.DataFrame.from_dict(losses).unstack().map(lambda x: x.item())
    data['ddim'] = config.ddata
    print(data)
    data.to_csv(os.path.join('runs', config.model_name, 'results', 'losses.csv'))
    writer.close()
    del data_vae
    del optimizer

if __name__ == "__main__":
    config_path = sys.argv[1]  # Read config_path from command line argument
    main(config_path=config_path)
