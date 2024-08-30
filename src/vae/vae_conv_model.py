import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from vae.vae_base import BaseVariationalAutoencoder, Sampling


class ConvEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super(ConvEncoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = feat_dim
        for i, num_filters in enumerate(hidden_layer_sizes):
            self.conv_layers.append(
                nn.Conv1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1)
            )
            in_channels = num_filters

        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim, hidden_layer_sizes)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.sampling = Sampling()

    def _get_last_dense_dim(self, seq_len, feat_dim, hidden_layer_sizes):
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            for conv in self.conv_layers:
                x = conv(x)
            return x.numel()

    def forward(self, x):
        x = x.transpose(1, 2) 
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.flatten(1)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
    

class ConvDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim):
        super(ConvDecoder, self).__init__()
        
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.dense = nn.Linear(latent_dim, encoder_last_dense_dim)
        self.deconv_layers = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]
        
        for i, num_filters in enumerate(reversed(hidden_layer_sizes[:-1])):
            self.deconv_layers.append(
                nn.ConvTranspose1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            in_channels = num_filters
            
        self.deconv_layers.append(
            nn.ConvTranspose1d(in_channels, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        L_in = encoder_last_dense_dim // hidden_layer_sizes[-1] 
        for i in range(len(hidden_layer_sizes)):
            L_in = (L_in - 1) * 2 - 2 * 1 + 3 + 1 
        L_final = L_in 

        self.final_dense = nn.Linear(feat_dim * L_final, seq_len * feat_dim)

    def forward(self, z):
        batch_size = z.size(0)
        x = F.relu(self.dense(z))
        x = x.view(batch_size, -1, self.hidden_layer_sizes[-1])
        x = x.transpose(1, 2)
        
        for deconv in self.deconv_layers[:-1]:
            x = F.relu(deconv(x))
        x = F.relu(self.deconv_layers[-1](x))
        
        x = x.flatten(1)
        x = self.final_dense(x)
        x = x.view(-1, self.seq_len, self.feat_dim)
        return x

class VariationalAutoencoderConv(BaseVariationalAutoencoder):
    model_name = "VAE_Conv"

    def __init__(self, hidden_layer_sizes=None, **kwargs):
        super(VariationalAutoencoderConv, self).__init__(**kwargs)
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_encoder(self):
        return ConvEncoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim)

    def _get_decoder(self):
        return ConvDecoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.encoder.encoder_last_dense_dim)

    @classmethod
    def load(cls, model_dir):
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = VariationalAutoencoderConv(**dict_params)
        vae_model.load_weights(model_dir)
        return vae_model