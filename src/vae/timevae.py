import torch
import torch.nn as nn
import torch.optim as optim

import os
import joblib

from vae.vae_base import BaseVariationalAutoencoder, Sampling


class TrendLayer(nn.Module):
    def __init__(self, latent_dim, feat_dim, trend_poly, seq_len):
        super(TrendLayer, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.seq_len = seq_len
        self.trend_dense1 = nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)
        self.relu = nn.ReLU()

    def forward(self, z):
        trend_params = self.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space = torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len 
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0) 

        trend_vals = torch.matmul(trend_params, poly_space) 
        trend_vals = trend_vals.permute(0, 2, 1) 
        trend_vals = trend_vals.float()

        return trend_vals


class SeasonalLayer(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len, custom_seas):
        super(SeasonalLayer, self).__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas

        self.dense_layers = nn.ModuleList([
            nn.Linear(latent_dim, feat_dim * num_seasons)
            for num_seasons, len_per_season in custom_seas
        ])
        

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[: self.seq_len]
        return season_indexes

    def forward(self, z):
        N = z.shape[0]
        ones_tensor = torch.ones((N, self.feat_dim, self.seq_len), dtype=torch.int32, device=z.device)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            ).to(z.device)

            dim2_idxes = ones_tensor * season_indexes_over_time.view(1, 1, -1)
            season_vals = torch.gather(season_params, 2, dim2_idxes)

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1) 
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)  
        all_seas_vals = all_seas_vals.permute(0, 2, 1)  

        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)
    

class LevelModel(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals


class ResidualConnection(nn.Module):
    def __init__(self, latent_dim, hidden_layer_sizes, feat_dim, seq_len, encoder_last_dense_dim):
        super(ResidualConnection, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.encoder_last_dense_dim = encoder_last_dense_dim
        
        self.linear1 = nn.Linear(latent_dim, encoder_last_dense_dim)
        self.relu = nn.ReLU()
        
        self.deconv_layers = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]
        for num_filters in reversed(hidden_layer_sizes[:-1]):
            self.deconv_layers.append(
                nn.ConvTranspose1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            in_channels = num_filters

        self.deconv_final = nn.ConvTranspose1d(hidden_layer_sizes[0], feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.flatten = nn.Flatten()
        self.linear2 = nn.Linear(seq_len * feat_dim, seq_len * feat_dim)

    def forward(self, z):
        batch_size = z.shape[0]
        x = self.linear1(z)
        x = self.relu(x)
        x = x.view(batch_size, self.hidden_layer_sizes[-1], -1)
        
        for deconv in self.deconv_layers:
            x = deconv(x)
            x = self.relu(x)

        x = self.deconv_final(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        x = self.linear2(x)
        residuals = x.view(batch_size, self.seq_len, self.feat_dim)
        return residuals
    

class TimeVAEEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_layer_sizes, latent_dim, seq_len):
        super(TimeVAEEncoder, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.layers = []
        self.layers.append(nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())

        for i, num_filters in enumerate(hidden_layer_sizes[1:]):
            self.layers.append(nn.Conv1d(hidden_layer_sizes[i], num_filters, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Flatten())
        
        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim, hidden_layer_sizes)

        self.encoder = nn.Sequential(*self.layers)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = Sampling()([z_mean, z_log_var])
        return z_mean, z_log_var, z
    
    def _get_last_dense_dim(self, seq_len, feat_dim, hidden_layer_sizes):
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            for conv in self.layers:
                x = conv(x)
            return x.numel()

class TimeVAEDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_layer_sizes, latent_dim, seq_len, trend_poly=0, custom_seas=None, use_residual_conn=True, encoder_last_dense_dim=None):
        super(TimeVAEDecoder, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder_last_dense_dim = encoder_last_dense_dim
        self.relu = nn.ReLU()
        self.level_model = LevelModel(self.latent_dim, self.feat_dim, self.seq_len)

        if use_residual_conn:
            self.residual_conn = ResidualConnection(latent_dim, hidden_layer_sizes, feat_dim, seq_len, encoder_last_dense_dim)


    def forward(self, z):
        outputs = self.level_model(z)
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = TrendLayer(self.latent_dim, self.feat_dim, self.trend_poly, self.seq_len)(z)
            outputs += trend_vals

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            cust_seas_vals = SeasonalLayer(self.latent_dim, self.feat_dim, self.seq_len, self.custom_seas)(z)
            outputs += cust_seas_vals

        if self.use_residual_conn:
            residuals = self.residual_conn(z)
            outputs += residuals

        return outputs


class TimeVAE(BaseVariationalAutoencoder):
    model_name = "TimeVAE"

    def __init__(
        self,
        hidden_layer_sizes=None,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        super(TimeVAE, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_encoder(self):
        return TimeVAEEncoder(self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.seq_len)

    def _get_decoder(self):
        return TimeVAEDecoder(self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.seq_len, self.trend_poly, self.custom_seas, self.use_residual_conn, self.encoder.encoder_last_dense_dim)

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f"{self.model_name}_weights.pth"))

        if self.custom_seas is not None:
            self.custom_seas = [(int(num_seasons), int(len_per_season)) for num_seasons, len_per_season in self.custom_seas]

        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "trend_poly": self.trend_poly,
            "custom_seas": self.custom_seas,
            "use_residual_conn": self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

    @classmethod
    def load(cls, model_dir: str) -> "TimeVAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = TimeVAE(**dict_params)
        vae_model.load_state_dict(torch.load(os.path.join(model_dir, f"{cls.model_name}_weights.pth")))
        return vae_model