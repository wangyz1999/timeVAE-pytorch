import os
import torch
import torch.nn as nn
import joblib
from vae.vae_base import BaseVariationalAutoencoder, Sampling

class TimeVAE(BaseVariationalAutoencoder):
    model_name = "TimeVAE"

    def __init__(
        self,
        hidden_layer_sizes,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        """
        hidden_layer_sizes: list of number of filters in convolutional layers in encoder and residual connection of decoder.
        trend_poly: integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term.
        custom_seas: list of tuples of (num_seasons, len_per_season).
            num_seasons: number of seasons per cycle.
            len_per_season: number of epochs (time-steps) per season.
        use_residual_conn: boolean value indicating whether to use a residual connection for reconstruction in addition to
        trend, generic and custom seasonalities.
        """

        super(TimeVAE, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        