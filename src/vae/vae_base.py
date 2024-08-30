import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib


class Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class BaseVariationalAutoencoder(nn.Module, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        reconstruction_wt=3.0,
        batch_size=16,
        **kwargs
    ):
        super(BaseVariationalAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.encoder = None
        self.decoder = None
        self.sampling = Sampling()

    def fit_on_data(self, train_data, max_epochs=1000, verbose=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        train_tensor = torch.FloatTensor(train_data).to(device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters())
        
        for epoch in range(max_epochs):
            self.train()
            total_loss = 0
            reconstruction_loss = 0
            kl_loss = 0
            
            for batch in train_loader:
                X = batch[0]
                optimizer.zero_grad()
                
                z_mean, z_log_var, z = self.encoder(X)
                reconstruction = self.decoder(z)
                
                loss, recon_loss, kl = self.loss_function(X, reconstruction, z_mean, z_log_var)
                
                # Normalize the loss by the batch size
                loss = loss / X.size(0)
                recon_loss = recon_loss / X.size(0)
                kl = kl / X.size(0)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                reconstruction_loss += recon_loss.item()
                kl_loss += kl.item()
            
            if verbose:
                print(f"Epoch {epoch + 1}/{max_epochs} | Total loss: {total_loss / len(train_loader):.4f} | "
                    f"Recon loss: {reconstruction_loss / len(train_loader):.4f} | "
                    f"KL loss: {kl_loss / len(train_loader):.4f}")

    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        return x_decoded
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(next(self.parameters()).device)
            z_mean, z_log_var, z = self.encoder(X)
            x_decoded = self.decoder(z_mean)
        return x_decoded.cpu().detach().numpy()

    def get_num_trainable_variables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_prior_samples(self, num_samples):
        device = next(self.parameters()).device
        Z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    def get_prior_samples_given_Z(self, Z):
        Z = torch.FloatTensor(Z).to(next(self.parameters()).device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_recons, dim):
            x_r = torch.mean(X, dim=dim)
            x_c_r = torch.mean(X_recons, dim=dim)
            err = torch.pow(x_r - x_c_r, 2)
            loss = torch.sum(err)
            return loss

        err = torch.pow(X - X_recons, 2)
        reconst_loss = torch.sum(err)
        
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=2)  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=1)  # by feature axis 

        return reconst_loss

    def loss_function(self, X, X_recons, z_mean, z_log_var):
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_decoder_wts.pth"))

    def load_weights(self, model_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth")))
        self.decoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_decoder_wts.pth")))

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes) if hasattr(self, 'hidden_layer_sizes') else None,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

if __name__ == "__main__":
    pass