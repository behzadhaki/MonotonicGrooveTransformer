#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu
import torch
import numpy as np
from model import GrooveTransformerEncoderVAE
from eval.GrooveEvaluator import load_evaluator_template

from logging import getLogger
logger = getLogger("helpers.VAE.eval_utils")
logger.setLevel("DEBUG")


def get_logging_media_for_vae_model_wandb(
        groove_transformer_vae, device, dataset_setting_json_path, subset_name,
        down_sampled_ratio, collapse_tapped_sequence,
        cached_folder="eval/GrooveEvaluator/templates/",
        divide_by_genre=True, **kwargs):
    """
    Prepare the media for logging in wandb. Can be easily used with an evaluator template
    (A template can be created using the code in eval/GrooveEvaluator/templates/main.py)
    :param groove_transformer_vae: The model to be evaluated
    :param device: The device to be used for evaluation
    :param dataset_setting_json_path: The path to the dataset setting json file
    :param subset_name: The name of the subset to be evaluated
    :param down_sampled_ratio: The ratio of the subset to be evaluated
    :param collapse_tapped_sequence: Whether to collapse the tapped sequence or not (input will have 1 voice only)
    :param cached_folder: The folder to be used for caching the evaluator template
    :param divide_by_genre: Whether to divide the subset by genre or not
    :param kwargs:                  additional arguments: need_hit_scores, need_velocity_distributions,
                                    need_offset_distributions, need_rhythmic_distances, need_heatmap
    :return:                        a ready to use dictionary to be logged using wandb.log()
    """

    # and model is correct type
    assert isinstance(groove_transformer_vae, GrooveTransformerEncoderVAE)

    # load the evaluator template (or create a new one if it does not exist)
    evaluator = load_evaluator_template(
        dataset_setting_json_path=dataset_setting_json_path,
        subset_name=subset_name,
        down_sampled_ratio=down_sampled_ratio,
        cached_folder=cached_folder,
        divide_by_genre=divide_by_genre
    )

    # logger.info("Generating the PianoRolls for subset: {}".format(subset_name))

    # Prepare the flags for require media
    # ----------------------------------
    need_hit_scores = kwargs["need_hit_scores"] if "need_hit_scores" in kwargs.keys() else False
    need_velocity_distributions = kwargs["need_velocity_distributions"] \
        if "need_velocity_distributions" in kwargs.keys() else False
    need_offset_distributions = kwargs["need_offset_distributions"] \
        if "need_offset_distributions" in kwargs.keys() else False
    need_rhythmic_distances = kwargs["need_rhythmic_distances"] \
        if "need_rhythmic_distances" in kwargs.keys() else False
    need_heatmap = kwargs["need_heatmap"] if "need_heatmap" in kwargs.keys() else False
    need_global_features = kwargs["need_global_features"] \
        if "need_global_features" in kwargs.keys() else False
    need_piano_roll = kwargs["need_piano_roll"] if "need_piano_roll" in kwargs.keys() else False
    need_audio = kwargs["need_audio"] if "need_audio" in kwargs.keys() else False
    need_kl_oa = kwargs["need_kl_oa"] if "need_kl_oa" in kwargs.keys() else False

    # (1) Get the targets, (2) tapify and pass to the model (3) add the predictions to the evaluator
    # ------------------------------------------------------------------------------------------
    hvo_seqs = evaluator.get_ground_truth_hvo_sequences()
    in_groove = torch.tensor(
        np.array([hvo_seq.flatten_voices(reduce_dim=collapse_tapped_sequence)
                  for hvo_seq in hvo_seqs]), dtype=torch.float32).to(
        device)
    hvos_array, _, _, _ = groove_transformer_vae.predict(in_groove, return_concatenated=True)
    evaluator.add_predictions(hvos_array.detach().cpu().numpy())

    # Get the media from the evaluator
    # -------------------------------
    media = evaluator.get_logging_media(
        prepare_for_wandb=True,
        need_hit_scores=need_hit_scores,
        need_velocity_distributions=need_velocity_distributions,
        need_offset_distributions=need_offset_distributions,
        need_rhythmic_distances=need_rhythmic_distances,
        need_heatmap=need_heatmap,
        need_global_features=need_global_features,
        need_piano_roll=need_piano_roll,
        need_audio=need_audio,
        need_kl_oa=need_kl_oa)

    return media


def get_hit_scores_for_vae_model(groove_transformer_vae, device, dataset_setting_json_path, subset_name,
                            down_sampled_ratio, collapse_tapped_sequence,
                                 cached_folder="eval/GrooveEvaluator/templates/",
                            divide_by_genre=True):

    # logger.info("Generating the hit scores for subset: {}".format(subset_name))
    # and model is correct type

    assert isinstance(groove_transformer_vae, GrooveTransformerEncoderVAE)

    # load the evaluator template (or create a new one if it does not exist)
    evaluator = load_evaluator_template(
        dataset_setting_json_path=dataset_setting_json_path,
        subset_name=subset_name,
        down_sampled_ratio=down_sampled_ratio,
        cached_folder=cached_folder,
        divide_by_genre=divide_by_genre
    )

    # (1) Get the targets, (2) tapify and pass to the model (3) add the predictions to the evaluator
    # ------------------------------------------------------------------------------------------
    hvo_seqs = evaluator.get_ground_truth_hvo_sequences()

    in_groove = torch.tensor(
        np.array([hvo_seq.flatten_voices(reduce_dim=collapse_tapped_sequence)
                  for hvo_seq in hvo_seqs]), dtype=torch.float32)
    predictions = []

    # batchify the input
    groove_transformer_vae.eval()
    with torch.no_grad():
        for batch_ix, batch_in in enumerate(torch.split(in_groove, 32)):
            hvos_array, _, _, _ = groove_transformer_vae.predict(
                batch_in.to(device),
                return_concatenated=True)
            predictions.append(hvos_array.detach().cpu().numpy())

    evaluator.add_predictions(np.concatenate(predictions))

    hit_dict = evaluator.get_statistics_of_pos_neg_hit_scores()

    score_dict = {f"{subset_name}/{key}_mean".replace(" ","_").replace("-","_"): float(value['mean']) for key, value in hit_dict.items()}
    score_dict.update({f"{subset_name}/{key}_std".replace(" ","_").replace("-","_"): float(value['std']) for key, value in hit_dict.items()})
    return score_dict
