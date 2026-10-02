"""
Dense autoencoder (PyTorch) trained on spec-normalized deviation features from
healthy operating periods. Reconstruction error = anomaly score. Catches
subtle joint patterns across many parameters that Isolation Forest's
tree-splits might miss — this is the "deep pattern" layer.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "autoencoder.pt"


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, bottleneck_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AnomalyDetectorAE:
    def __init__(self, input_dim, hidden_dim=12, bottleneck_dim=4, lr=1e-3):
        self.model = DenseAutoencoder(input_dim, hidden_dim, bottleneck_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction="none")
        self.input_dim = input_dim
        self.error_scale = 1.0  # set during fit, used to normalize scores to ~0-1

    def fit(self, X, epochs=100, batch_size=32, verbose=False):
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        X_t = torch.tensor(X)
        n = X_t.shape[0]

        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch = X_t[idx]
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.loss_fn(recon, batch).mean()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(idx)
            if verbose and epoch % 20 == 0:
                print(f"epoch {epoch}, loss {epoch_loss/n:.5f}")

        # establish error_scale from training reconstruction errors (95th percentile)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_t)
            errors = self.loss_fn(recon, X_t).mean(dim=1).numpy()
        self.error_scale = max(np.percentile(errors, 95), 1e-6)
        return self

    def score(self, X):
        """Returns reconstruction-error-based anomaly score, normalized ~0-1 (can exceed 1 for extreme anomalies)."""
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        X_t = torch.tensor(X)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_t)
            errors = self.loss_fn(recon, X_t).mean(dim=1).numpy()
        return np.clip(errors / self.error_scale, 0, None)

    def save(self, path=MODEL_PATH):
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "error_scale": self.error_scale,
        }, path)

    def load(self, path=MODEL_PATH):
        checkpoint = torch.load(path, weights_only=False)
        self.input_dim = checkpoint["input_dim"]
        self.model = DenseAutoencoder(self.input_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.error_scale = checkpoint["error_scale"]
        return self
