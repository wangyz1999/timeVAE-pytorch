import torch
import torch.nn as nn
import torch.optim as optim

import os
import joblib

from vae_base import BaseVariationalAutoencoder, Sampling


class TrendLayer(nn.Module):
    def __init__(self, feat_dim, trend_poly, seq_len):
        super(TrendLayer, self).__init__()
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.seq_len = seq_len
        self.trend_dense1 = nn.Linear(self.feat_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)
        self.relu = nn.ReLU()

    def forward(self, z):
        trend_params = self.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)  # shape: N x D x P

        lin_space = torch.arange(0, float(self.seq_len), 1) / self.seq_len  # shape of lin_space: 1d tensor of length T
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0)  # shape: P x T

        trend_vals = torch.matmul(trend_params, poly_space)  # shape (N, D, T)
        trend_vals = trend_vals.permute(0, 2, 1)  # shape: (N, T, D)
        trend_vals = trend_vals.float()

        return trend_vals


class SeasonalLayer(nn.Module):
    def __init__(self, feat_dim, seq_len, custom_seas):
        super(SeasonalLayer, self).__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas

        # Create dense layers for each season
        self.dense_layers = nn.ModuleList([
            nn.Linear(feat_dim, feat_dim * num_seasons)
            for num_seasons, len_per_season in custom_seas
        ])
        

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        # Ensure the length matches seq_len
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[: self.seq_len]
        return season_indexes

    def forward(self, z):
        N = z.shape[0]
        ones_tensor = torch.ones((N, self.feat_dim, self.seq_len), dtype=torch.int32)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)  # shape: (N, D * S)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)  # shape: (N, D, S)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            )  # shape: (T, )

            dim2_idxes = ones_tensor * season_indexes_over_time.view(1, 1, -1)  # shape: (N, D, T)
            season_vals = torch.gather(season_params, 2, dim2_idxes)  # shape (N, D, T)

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1)  # shape: (N, D, T, S)
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)  # shape (N, D, T)
        all_seas_vals = all_seas_vals.permute(0, 2, 1)  # shape (N, T, D)

        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)


class TimeVAEEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_layer_sizes, latent_dim, seq_len):
        super(TimeVAEEncoder, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        layers = []
        layers.append(nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())

        for i, num_filters in enumerate(hidden_layer_sizes[1:]):
            layers.append(nn.Conv1d(hidden_layer_sizes[i], num_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())

        layers.append(nn.Flatten())
        
        # Calculate the size after flattening
        self.encoder_last_dense_dim = hidden_layer_sizes[-1] * (seq_len // 2**len(hidden_layer_sizes))

        self.encoder = nn.Sequential(*layers)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


class TimeVAEDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_layer_sizes, latent_dim, seq_len, trend_poly=0, custom_seas=None, use_residual_conn=True):
        super(TimeVAEDecoder, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn

        self.relu = nn.ReLU()

        # Transposed convolutions for upsampling
        self.deconv_layers = nn.ModuleList()
        for i, num_filters in enumerate(reversed(hidden_layer_sizes[:-1])):
            self.deconv_layers.append(nn.ConvTranspose1d(num_filters, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1))
            self.deconv_layers.append(nn.ReLU())

        self.final_deconv = nn.ConvTranspose1d(hidden_layer_sizes[0], feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_relu = nn.ReLU()

    def forward(self, z):
        outputs = self.level_model(z)
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = TrendLayer(self.feat_dim, self.trend_poly, self.seq_len)(z)
            outputs += trend_vals

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            cust_seas_vals = SeasonalLayer(self.feat_dim, self.seq_len, self.custom_seas)(z)
            outputs += cust_seas_vals

        if self.use_residual_conn:
            residuals = self._get_decoder_residual(z)
            outputs += residuals

        return outputs

    def _get_decoder_residual(self, z):
        x = nn.Linear(self.latent_dim, self.hidden_layer_sizes[-1])(z)
        x = nn.ReLU()(x)
        x = x.view(-1, self.hidden_layer_sizes[-1], self.seq_len // 2**len(self.hidden_layer_sizes))

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = nn.ConvTranspose1d(num_filters, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)(x)
            x = nn.ReLU()(x)

        x = nn.ConvTranspose1d(self.hidden_layer_sizes[0], self.feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)(x)
        x = nn.ReLU()(x)

        x = nn.Flatten()(x)
        x = nn.Linear(self.seq_len * self.feat_dim, self.seq_len * self.feat_dim)(x)
        residuals = x.view(-1, self.seq_len, self.feat_dim)
        return residuals
    
    def level_model(self, z):
        level_params = nn.Linear(self.feat_dim, self.feat_dim)(z)
        level_params = nn.ReLU()(level_params)
        level_params = nn.Linear(self.feat_dim, self.feat_dim)(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32)
        level_vals = level_params * ones_tensor
        return level_vals


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

        self.optimizer = optim.Adam(self.parameters())

    def _get_encoder(self):
        return TimeVAEEncoder(self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.seq_len)

    def _get_decoder(self):
        return TimeVAEDecoder(self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.seq_len, self.trend_poly, self.custom_seas, self.use_residual_conn)

    

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f"{self.model_name}_weights.pth"))

        # Convert custom_seas back to list of tuples so it is serializable
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
        vae_model.optimizer = optim.Adam(vae_model.parameters())
        return vae_model


if __name__ == "__main__":
    # Test parameters
    seq_len = 128
    feat_dim = 16
    latent_dim = 8
    hidden_layer_sizes = [32, 64, 128]
    trend_poly = 2
    custom_seas = [(4, 32), (6, 16)]
    use_residual_conn = True

    # Initialize the TimeVAE model
    vae_model = TimeVAE(
        hidden_layer_sizes=hidden_layer_sizes,
        trend_poly=trend_poly,
        custom_seas=custom_seas,
        use_residual_conn=use_residual_conn,
        seq_len=seq_len,
        feat_dim=feat_dim,
        latent_dim=latent_dim,
    )

    # Print the model architecture
    print("Encoder Architecture:")
    print(vae_model.encoder)

    print("\nDecoder Architecture:")
    print(vae_model.decoder)

    # Create a random input tensor with shape (batch_size, seq_len, feat_dim)
    batch_size = 4
    input_tensor = torch.randn(batch_size, feat_dim, seq_len)

    # Process the input tensor through the encoder
    z_mean, z_log_var = vae_model.encoder(input_tensor)

    print("\nProcessed tensor (z_mean):")
    print(z_mean)

    print("\nProcessed tensor (z_log_var):")
    print(z_log_var)

    # Test the decoder with a random latent tensor
    latent_tensor = torch.randn(batch_size, latent_dim)
    reconstructed_tensor = vae_model.decoder(latent_tensor)

    print("\nReconstructed tensor:")
    print(reconstructed_tensor)
