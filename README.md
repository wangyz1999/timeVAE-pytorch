# PyTorch Implementation of TimeVAE

This repository provides an unofficial PyTorch implementation of the TimeVAE model for generating synthetic time-series data, along with two baseline models: a dense VAE and a convolutional VAE. The file structures and usage closely follow the original TensorFlow implementation to ensure consistency and ease of use.

Original Tensorflow Repo: [TimeVAE for Synthetic Timeseries Data Generation](https://github.com/abudesai/timeVAE)

## Paper Reference

For a detailed explanation of the methodology, please refer to the original paper: [TIMEVAE: A VARIATIONAL AUTO-ENCODER FOR MULTIVARIATE TIME SERIES GENERATION](https://arxiv.org/abs/2111.08095).

## Comparison 

The PyTorch model was trained and evaluated using the provided dataset and the default hyperparameters as in the TensorFlow implementation, each for 1000 epochs, achieving similar convergence (see Figure 4 from the original paper). The plotting script can be found in [src/compare_plot.ipynb](https://github.com/wangyz1999/timeVAE-pytorch/blob/main/src/compare_plot.ipynb).

![TSNE TIMEVAE](https://github.com/user-attachments/assets/887a776c-7df6-46f4-9a16-301eb6021967)


## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

1. **Prepare Data**: Save your data as a numpy array with shape `(n_samples, n_timesteps, n_features)` in the `./data/` folder in `.npz` format. The filename without the extension will be used as the dataset name (e.g., `my_data.npz` will be referred to as `my_data`). Alternatively, use one of the existing datasets provided in the `./data/` folder.

2. **Configure Pipeline**:

   - Update the dataset name and model type in `./src/vae_pipeline.py`:
     ```python
     dataset = "my_data"  # Your dataset name
     model_name = "timeVAE"  # Choose between vae_dense, vae_conv, or timeVAE
     ```
   - Set hyperparameters in `./src/config/hyperparameters.yaml`. Key hyperparameters include `latent_dim`, `hidden_layer_sizes`, `reconstruction_wt`, and `batch_size`.

3. **Run the Script**:

   ```bash
   python src/vae_pipeline.py
   ```

4. **Outputs**:
   - Trained models are saved in `./outputs/models/<dataset_name>/`.
   - Generated synthetic data is saved in `./outputs/gen_data/<dataset_name>/` in `.npz` format.
   - t-SNE plots are saved in `./outputs/tsne/<dataset_name>/` in `.png` format.

## Hyperparameters

The four key hyperparameters for the VAE models are:

- `latent_dim`: Number of latent dimensions (default: 8).
- `hidden_layer_sizes`: Number of hidden units or filters (default: [50, 100, 200]).
- `reconstruction_wt`: Weight for the reconstruction loss (default: 3.0).
- `batch_size`: Training batch size (default: 16).

For `timeVAE`:

- `trend_poly`: Degree of polynomial trend component (default: 0).
- `custom_seas`: Custom seasonalities as a list of tuples (default: null).
- `use_residual_conn`: Use residual connection (default: true).

> The default settings for the timeVAE model set it to operate as the base model without interpretable components.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
