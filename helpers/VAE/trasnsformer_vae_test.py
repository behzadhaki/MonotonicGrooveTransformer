#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
from model import GrooveTransformerEncoderVAE

config = {
    'd_model_enc': 128,
    'd_model_dec': 512,
    'embedding_size_src': 9,
    'embedding_size_tgt': 27,
    'nhead_enc': 2,
    'nhead_dec': 4,
    'dim_feedforward_enc': 16,
    'dim_feedforward_dec': 32,
    'num_encoder_layers': 3,
    'num_decoder_layers': 5,
    'dropout': 0.1,
    'latent_dim': 32,
    'max_len_enc': 32,
    'max_len_dec': 32,
    'device': 'cpu',
    'o_activation': 'sigmoid'
}

src = torch.rand(20, config["max_len_enc"], config["embedding_size_src"])

### call VAE encoder

encoVae = GrooveTransformerEncoderVAE(config)

(h_logits, v_logits, o_logits), mu, log_var = encoVae(src)

print(f"h_logits: {h_logits.shape}, v_logits: {v_logits.shape}, "
      f"o_logits: {o_logits.shape}, mu: {mu.shape}, log_var: {log_var.shape}")

hvo, mu, log_var = encoVae.predict(src)