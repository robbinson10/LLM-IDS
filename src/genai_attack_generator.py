import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/genai/rare_attacks.csv"
OUTPUT_PATH = "data/genai/synthetic_attacks.csv"
EPOCHS = 50
BATCH_SIZE = 8
LATENT_DIM = 8

# -----------------------------
# VAE Model
# -----------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# -----------------------------
# Loss Function
# -----------------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading rare attack dataset...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    labels = df["Label"]
    X = df.drop(columns=["Label"])

    print("Rare dataset shape:", X.shape)
    print("Classes:\n", labels.value_counts())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    input_dim = X_tensor.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining VAE...")
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(X_tensor.size(0))

        epoch_loss = 0
        for i in range(0, X_tensor.size(0), BATCH_SIZE):
            batch_idx = perm[i:i+BATCH_SIZE]
            batch = X_tensor[batch_idx]

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

    # -----------------------------
    # Generate Synthetic Samples
    # -----------------------------
    print("\nGenerating synthetic attack samples...")
    num_samples = 500   # You can increase later
    z = torch.randn(num_samples, LATENT_DIM)
    synthetic_data = model.decode(z).detach().numpy()

    # Inverse scale
    synthetic_data = scaler.inverse_transform(synthetic_data)

    synthetic_df = pd.DataFrame(synthetic_data, columns=X.columns)
    synthetic_df["Label"] = "Synthetic_Attack"

    os.makedirs("data/genai", exist_ok=True)
    synthetic_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Synthetic attack data saved to: {OUTPUT_PATH}")
    print("GenAI synthetic attack generation complete.")

if __name__ == "__main__":
    main()
