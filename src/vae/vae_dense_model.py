import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib

from vae.vae_base import BaseVariationalAutoencoder, Sampling

class DenseEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super(DenseEncoder, self).__init__()
        input_size = seq_len * feat_dim

        encoder_layers = []
        
        encoder_layers.append(nn.Flatten())
        
        for M_out in hidden_layer_sizes:
            encoder_layers.append(nn.Linear(input_size, M_out))
            encoder_layers.append(nn.ReLU())
            input_size = M_out

        self.encoder = nn.Sequential(*encoder_layers)
        self.z_mean = nn.Linear(input_size, latent_dim)
        self.z_log_var = nn.Linear(input_size, latent_dim)
        self.sampling = Sampling()
    
    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class DenseDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super(DenseDecoder, self).__init__()
        decoder_layers = []
        input_size = latent_dim
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        
        for M_out in hidden_layer_sizes:
            decoder_layers.append(nn.Linear(input_size, M_out))
            decoder_layers.append(nn.ReLU())
            input_size = M_out
        
        decoder_layers.append(nn.Linear(input_size, seq_len * feat_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, z):
        decoder_output = self.decoder(z)
        reshaped_output = decoder_output.view(-1, self.seq_len, self.feat_dim)
        return reshaped_output

class VariationalAutoencoderDense(BaseVariationalAutoencoder):
    model_name = "VAE_Dense"

    def __init__(self, hidden_layer_sizes, **kwargs):
        super(VariationalAutoencoderDense, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        self.optimizer = optim.Adam(self.parameters())

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_encoder(self):
        return DenseEncoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim)
    
    def _get_decoder(self):
        return DenseDecoder(self.seq_len, self.feat_dim, list(reversed(self.hidden_layer_sizes)), self.latent_dim)

    @classmethod
    def load(cls, model_dir: str) -> "VariationalAutoencoderDense":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = VariationalAutoencoderDense(**dict_params)
        vae_model.load_state_dict(torch.load(os.path.join(model_dir, f"{cls.model_name}_weights.pth")))
        vae_model.optimizer = optim.Adam(vae_model.parameters())
        return vae_model

    def save_weights(self, model_dir):
        torch.save(self.state_dict(), os.path.join(model_dir, f"{self.model_name}_weights.pth"))
