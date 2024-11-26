import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from vae_llf.loaders import load_config, ADNIDataset
from vae_llf.models import DenseEncoder, VAEDecoder, GenVAE  # Adjust based on your model choice

def main(config_path: str):
    # Load configuration
    config = load_config(config_path)

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # Initialize dataset and dataloader
    dataset = ADNIDataset(DBDIR='path/to/data', config=config)  # Adjust DBDIR as needed
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model
    encoder = DenseEncoder(input_dim=dataset.dataframe.shape[1], intermediate_dim=config.interm_dim, latent_dim=config.ddata).to(device)
    decoder = VAEDecoder(latent_dim=config.ddata).to(device)  # Adjust based on your model choice

    # Define optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config.lr)

    # Training loop
    results = []
    for epoch in range(config.n_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            # Forward pass
            z, mu, logvar = encoder(batch.to(device))
            recon_batch = decoder(z)

            # Compute loss (you may need to define your loss function)
            loss = compute_loss(batch.to(device), recon_batch, mu, logvar)  # Define compute_loss function

            # Backward pass
            loss.backward()
            optimizer.step()

            results.append({'epoch': epoch, 'loss': loss.item()})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('training_results.csv', index=False)

if __name__ == "__main__":
    config_path = sys.argv[1]  # Read config_path from command line argument
    main(config_path=config_path)
