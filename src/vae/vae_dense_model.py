import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib

from vae.vae_base import BaseVariationalAutoencoder, Sampling

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

    def _get_encoder(self):
        encoder_layers = []
        input_size = self.seq_len * self.feat_dim
        
        encoder_layers.append(nn.Flatten())
        
        for i, M_out in enumerate(self.hidden_layer_sizes):
            encoder_layers.append(nn.Linear(input_size, M_out))
            encoder_layers.append(nn.ReLU())
            input_size = M_out

        self.z_mean = nn.Linear(input_size, self.latent_dim)
        self.z_log_var = nn.Linear(input_size, self.latent_dim)
        
        return nn.Sequential(*encoder_layers)

    def _get_decoder(self):
        decoder_layers = []
        input_size = self.latent_dim
        
        for i, M_out in enumerate(reversed(self.hidden_layer_sizes)):
            decoder_layers.append(nn.Linear(input_size, M_out))
            decoder_layers.append(nn.ReLU())
            input_size = M_out
        
        decoder_layers.append(nn.Linear(input_size, self.seq_len * self.feat_dim))
        decoder_layers.append(nn.Unflatten(1, (self.seq_len, self.feat_dim)))
        
        return nn.Sequential(*decoder_layers)

    def encode(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = Sampling()([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var, z = self.encode(x)
        return self.decode(z), z_mean, z_log_var

    def compile(self, optimizer):
        self.optimizer = optimizer

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