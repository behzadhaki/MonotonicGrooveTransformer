import math
import yaml

sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'loss',
                'goal': 'minimize'          # FIXME: Maximize accuracy
            }}

parameters_dict = {
            'nhead_enc': {
                        'values': [1, 2, 4, 8, 16]},
            'nhead_dec': {
                        'values': [1, 2, 4, 8, 16]},
            'd_model_enc': {
                        'values': [16, 32, 64, 128, 256, 512]},
            'd_model_dec': {
                        'values': [16, 32, 64, 128, 256, 512]},
            'embedding_size_src': {
                        'value': 27},
            'embedding_size_tgt': {
                        'value': 27},
            'dim_feedforward': {
                        'values': [16, 32, 64, 128, 256, 512]},
            'dropout': {
                 'distribution': 'uniform',
                 'min': 0.1,
                 'max': 0.3
            },
            'loss_hit_penalty_multiplier': {
                 'distribution': 'uniform',
                 'min': 0,
                 'max': 1
            },
            'num_encoder_layers': {
                        'values': [6, 8, 10, 12]},
            'num_decoder_layers': {
                        'values': [6, 8, 10, 12]},
            'max_len': {
                    'value': 32},
            'device': {
                    'value': 0},
            'latent_dim': {
                    'value': int((32 * 27) / 4)},
            "epochs": {
                    'value': 100},
            "batch_size": {
                    'values': [16, 32]},
            "lr": {
                    'values': [1e-3, 1e-4]},
            "bce": {
                'values': [True , False]},
            "dice": {
                'values': [True , False]},
            }

sweep_config['parameters'] = parameters_dict



with open('model_params.yaml', 'w') as outfile:
    yaml.dump(sweep_config, outfile, default_flow_style=False)