model_params = {
    'light_version':
        {
            'd_model': 128,
            'embedding_sz': 27,
            'n_heads': 4,
            'dim_ff': 128,
            'dropout': 0.1038,
            'n_layers': 11,
            'max_len': 32,
            'device': 'cpu' },
    'heavy_version':
        {
            'd_model': 512,
            'embedding_sz': 27,
            'n_heads': 4,
            'dim_ff': 16,
            'dropout': 0.1093,
            'n_layers': 6,
            'max_len': 32,
            'device': 'cpu'
        }
}