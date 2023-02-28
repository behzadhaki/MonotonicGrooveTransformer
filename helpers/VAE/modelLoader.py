#  Copyright (c) 2022. \n Created by Hernan Dario Perez
import torch
from model.VAE.MonotonicGrooveVAE import GrooveTransformerEncoderVAE
from logging import getLogger
logger = getLogger("helpers/VAE/modelLoader.py")
logger.setLevel("DEBUG")


def load_variational_mgt_model(model_path, params_dict=None, is_evaluating=True, device=None):
    """ Load a GrooveTransformerEncoder model from a given path

    Args:
        model_path (str): path to the model
        params_dict (None, dict, or json path): dictionary containing the parameters of the model
        is_evaluating (bool): if True, the model is set to eval mode
        device (None or torch.device): device to load the model to (if cpu, the model is loaded to cpu)

    Returns:
        model (GrooveTransformerEncoder): the loaded model
    """

    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device)
        else:
            loaded_dict = torch.load(model_path)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))
        logger.info(f"Model was loaded to cpu!!!")

    if params_dict is None:
        if 'params' in loaded_dict:
            params_dict = loaded_dict['params']
        else:
            raise Exception(f"Could not instantiate model as params_dict is not found. "
                            f"Please provide a params_dict either as a json path or as a dictionary")

    if isinstance(params_dict, str):
        import json
        with open(params_dict, 'r') as f:
            params_dict = json.load(f)

    model = GrooveTransformerEncoderVAE(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if is_evaluating:
        model.eval()

    return model
