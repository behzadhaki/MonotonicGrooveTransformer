# Import model
import torch
from trained_torch_models.params import model_params
from BaseGrooveTransformers.models.transformer import GrooveTransformerEncoder


def load_model(model_name, model_path):

    # load model parameters from params.py file
    params = model_params[model_name]

    # load checkpoint
    checkpoint = torch.load(model_path, map_location=params['device'])

    # Initialize model
    groove_transformer = GrooveTransformerEncoder(params['d_model'],
                                                  params['embedding_sz'],
                                                  params['embedding_sz'],
                                                  params['n_heads'],
                                                  params['dim_ff'],
                                                  params['dropout'],
                                                  params['n_layers'],
                                                  params['max_len'],
                                                  params['device'])

    # Load model and put in evaluation mode
    groove_transformer.load_state_dict(checkpoint['model_state_dict'])
    groove_transformer.eval()

    return groove_transformer

def place_note_in_tensor():
    pass