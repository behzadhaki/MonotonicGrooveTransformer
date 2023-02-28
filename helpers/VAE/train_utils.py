
import os
import torch
#from torchmetrics import Accuracy
import re
import numpy as np
from model.Base.BasicGrooveTransformer import GrooveTransformerEncoder, GrooveTransformer

from logging import getLogger
logger = getLogger("VAE_LOSS_CALCULATOR")
logger.setLevel("DEBUG")


def calculate_hit_loss(hit_logits, hit_targets, hit_loss_function):
    """
    Calculate the hit loss for the given hit logits and hit targets.
    The loss is calculated either using BCE or Dice loss function.
    :param hit_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param hit_targets:     (torch.Tensor)  target output of the model
    :param hit_loss_function:     (torch.nn.BCEWithLogitsLoss)
    :return:    hit_loss (batch, time_steps, n_voices)  the hit loss value per each step and voice (unreduced)
    """
    assert isinstance(hit_loss_function, torch.nn.BCEWithLogitsLoss)
    loss_h = hit_loss_function(hit_logits, hit_targets)           # batch, time steps, voices
    return loss_h       # batch_size,  time_steps, n_voices


def calculate_velocity_loss(vel_logits, vel_targets, vel_loss_function):
    """
    Calculate the velocity loss for the velocity targets and the **Pre-Activation** output of the model.
    The loss is calculated using either MSE or BCE loss function.
    :param vel_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param vel_targets:     (torch.Tensor)  target output of the model
    :param vel_loss_function:     (str)  either "mse" or "bce"
    :return:    vel_loss (batch_size, time_steps, n_voices)  the velocity loss value per each step and voice (unreduced)
    """
    if isinstance(vel_loss_function, torch.nn.MSELoss):
        loss_v = vel_loss_function(torch.sigmoid(vel_logits), vel_targets)
    elif isinstance(vel_loss_function, torch.nn.BCEWithLogitsLoss):
        loss_v = vel_loss_function(vel_logits, vel_targets)
    else:
        raise NotImplementedError(f"the vel_loss_function {vel_loss_function} is not implemented")

    return loss_v       # batch_size,  time_steps, n_voices


def calculate_offset_loss(offset_logits, offset_targets, offset_loss_function):
    """
    Calculate the offset loss for the offset targets and the **Pre-Activation** output of the model.
    The loss is calculated using either MSE or BCE loss function.

    **For MSE, the offset_logit is first mapped to -0.5 to 0.5 using a tanh function. Alternatively, for BCE,
    it is assumed that the offset_logit will be activated using a sigmoid function.**

    :param offset_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param offset_targets:     (torch.Tensor)  target output of the model
    :param offset_loss_function:     (str)  either "mse" or "bce"
    :return:    offset_loss (batch_size, time_steps, n_voices)  the offset loss value per each step
                    and voice (unreduced)

    """

    if isinstance(offset_loss_function, torch.nn.MSELoss):
        # the offset logits after the tanh activation are in the range of -1 to 1 . Therefore, we need to
        # scale the offset targets to the same range. This is done by multiplying the offset values after
        # the tanh activation by 0.5
        loss_o = offset_loss_function(torch.tanh(offset_logits)*0.5, offset_targets)
    elif isinstance(offset_loss_function, torch.nn.BCEWithLogitsLoss):
        # here the offsets MUST be in the range of [0, 1]. Our existing targets are from [-0.5, 0.5].
        # So we need to shift them to [0, 1] range by adding 0.5
        loss_o = offset_loss_function(offset_logits, offset_targets+0.5)
    else:
        raise NotImplementedError(f"the offset_loss_function {offset_loss_function} is not implemented")

    return loss_o           # batch_size,  time_steps, n_voices


def calculate_kld_loss(mu, log_var):
    """ calculate the KLD loss for the given mu and log_var values against a standard normal distribution
    :param mu:  (torch.Tensor)  the mean values of the latent space
    :param log_var: (torch.Tensor)  the log variance values of the latent space
    :return:    kld_loss (torch.Tensor)  the KLD loss value (unreduced) shape: (batch_size,  time_steps, n_voices)

    """
    mu = mu.view(mu.shape[0], -1)
    log_var = log_var.view(log_var.shape[0], -1)
    kld_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

    return kld_loss     # batch_size,  time_steps, n_voices


def batch_loop(dataloader_, groove_transformer_vae, hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, device, optimizer=None, starting_step=None, kl_beta=1.0):
    """
    This function iteratively loops over the given dataloader and calculates the loss for each batch. If an optimizer is
    provided, it will also perform the backward pass and update the model parameters. The loss values are accumulated
    and returned at the end of the loop.

    **Can be used for both training and testing. In testing however, backpropagation will not be performed**


    :param dataloader_:     (torch.utils.data.DataLoader)  dataloader for the dataset
    :param groove_transformer_vae:  (GrooveTransformerVAE)  the model
    :param hit_loss_fn:     (str)  either "dice" or "bce"
    :param velocity_loss_fn:    (str)  either "mse" or "bce"
    :param offset_loss_fn:  (str)  either "mse" or "bce"
    :param device:  (torch.device)  the device to use for the model
    :param optimizer:   (torch.optim.Optimizer)  the optimizer to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KLD loss
    :return:    (dict)  a dictionary containing the loss values for the current batch

                metrics = {
                    "loss_total": np.mean(loss_total),
                    "loss_h": np.mean(loss_h),
                    "loss_v": np.mean(loss_v),
                    "loss_o": np.mean(loss_o),
                    "loss_KL": np.mean(loss_KL)}

                (int)  the current step of the optimizer (if provided)
    """
    # Prepare the metric trackers for the new epoch
    # ------------------------------------------------------------------------------------------
    loss_total, loss_recon, loss_h, loss_v, loss_o, loss_KL = [], [], [], [], [], []

    # Iterate over batches
    # ------------------------------------------------------------------------------------------
    for batch_count, (inputs_, outputs_,
                      hit_balancing_weights_per_sample_, genre_balancing_weights_per_sample_,
                      indices) in enumerate(dataloader_):
        # Move data to GPU if available
        # ---------------------------------------------------------------------------------------
        inputs = inputs_.to(device) if inputs_.device.type!= device else inputs_
        outputs = outputs_.to(device) if outputs_.device.type!= device else outputs_
        hit_balancing_weights_per_sample = hit_balancing_weights_per_sample_.to(device) \
            if hit_balancing_weights_per_sample_.device.type!= device else hit_balancing_weights_per_sample_
        genre_balancing_weights_per_sample = genre_balancing_weights_per_sample_.to(device) \
            if genre_balancing_weights_per_sample_.device.type!= device else genre_balancing_weights_per_sample_

        # Forward pass
        # ---------------------------------------------------------------------------------------
        (h_logits, v_logits, o_logits), mu, log_var, latent_z = groove_transformer_vae.forward(inputs)

        # Prepare targets for loss calculation
        h_targets, v_targets, o_targets = torch.split(outputs, int(outputs.shape[2] / 3), 2)

        # Compute losses
        # ---------------------------------------------------------------------------------------
        batch_loss_h = calculate_hit_loss(
            hit_logits = h_logits, hit_targets=h_targets, hit_loss_function=hit_loss_fn)
        batch_loss_h = (batch_loss_h * hit_balancing_weights_per_sample * genre_balancing_weights_per_sample).mean()

        batch_loss_v = calculate_velocity_loss(
            vel_logits=v_logits, vel_targets=v_targets, vel_loss_function=velocity_loss_fn)
        batch_loss_v = (batch_loss_v * hit_balancing_weights_per_sample * genre_balancing_weights_per_sample).mean()

        batch_loss_o = calculate_offset_loss(
            offset_logits=o_logits, offset_targets=o_targets, offset_loss_function=offset_loss_fn)
        batch_loss_o = (batch_loss_o * hit_balancing_weights_per_sample * genre_balancing_weights_per_sample).mean()

        batch_loss_KL = kl_beta * calculate_kld_loss(mu, log_var)
        batch_loss_KL = (batch_loss_KL * genre_balancing_weights_per_sample[:, 0, 0].view(-1, 1)).mean()

        batch_loss_recon = (batch_loss_h + batch_loss_v + batch_loss_o)
        batch_loss_total = (batch_loss_recon + batch_loss_KL)

        # Backpropagation and optimization step (if training)
        # ---------------------------------------------------------------------------------------
        if optimizer is not None:
            optimizer.zero_grad()
            batch_loss_total.backward()
            optimizer.step()

        # Update the per batch loss trackers
        # -----------------------------------------------------------------
        loss_h.append(batch_loss_h.item())
        loss_v.append(batch_loss_v.item())
        loss_o.append(batch_loss_o.item())
        loss_total.append(batch_loss_total.item())
        loss_recon.append(batch_loss_recon.item())
        loss_KL.append(batch_loss_KL.item())

        # Increment the step counter
        # ---------------------------------------------------------------------------------------
        if starting_step is not None:
            starting_step += 1

    # empty gpu cache if cuda
    if device != 'cpu':
        torch.cuda.empty_cache()

    metrics = {
        "loss_total": np.mean(loss_total),
        "loss_h": np.mean(loss_h),
        "loss_v": np.mean(loss_v),
        "loss_o": np.mean(loss_o),
        "loss_KL": np.mean(loss_KL),
        "loss_recon": np.mean(loss_recon)
    }

    if starting_step is not None:
        return metrics, starting_step
    else:
        return metrics


def train_loop(train_dataloader, groove_transformer_vae, optimizer, hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, device, starting_step, kl_beta=1):
    """
    This function performs the training loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward and backward pass for each batch. The loss values are accumulated and the average is
    returned at the end of the loop.

    :param train_dataloader:    (torch.utils.data.DataLoader)  dataloader for the training dataset
    :param groove_transformer_vae:  (GrooveTransformerVAE)  the model
    :param optimizer:  (torch.optim.Optimizer)  the optimizer to use for the model (sgd or adam)
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:  (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:      (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param loss_hit_penalty_multiplier:  (float)  the hit loss penalty multiplier
    :param device:  (str)  the device to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KL loss

    :return:    (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "train/loss_total": np.mean(loss_total),
                    "train/loss_h": np.mean(loss_h),
                    "train/loss_v": np.mean(loss_v),
                    "train/loss_o": np.mean(loss_o),
                    "train/loss_KL": np.mean(loss_KL)}
    """
    # ensure model is in training mode
    if groove_transformer_vae.training is False:
        logger.warning("Model is not in training mode. Setting to training mode.")
        groove_transformer_vae.train()

    # Run the batch loop
    metrics, starting_step = batch_loop(
        dataloader_=train_dataloader,
        groove_transformer_vae=groove_transformer_vae,
        hit_loss_fn=hit_loss_fn,
        velocity_loss_fn=velocity_loss_fn,
        offset_loss_fn=offset_loss_fn,
        device=device,
        optimizer=optimizer,
        starting_step=starting_step,
        kl_beta=kl_beta)

    metrics = {f"train/{key}": value for key, value in metrics.items()}
    return metrics, starting_step


def test_loop(test_dataloader, groove_transformer_vae, hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, device, kl_beta=1):
    """
    This function performs the test loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward pass for each batch. The loss values are accumulated and the average is returned at the end
    of the loop.

    :param test_dataloader:   (torch.utils.data.DataLoader)  dataloader for the test dataset
    :param groove_transformer_vae:  (GrooveTransformerVAE)  the model
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:    (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:    (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param loss_hit_penalty_multiplier:     (float)  the hit loss penalty multiplier
    :param device:  (str)  the device to use for the model
    :return:   (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "test/loss_total": np.mean(loss_total),
                    "test/loss_h": np.mean(loss_h),
                    "test/loss_v": np.mean(loss_v),
                    "test/loss_o": np.mean(loss_o),
                    "test/loss_KL": np.mean(loss_KL)}
    """
    # ensure model is in eval mode
    if groove_transformer_vae.training is True:
        logger.warning("Model is not in eval mode. Setting to eval mode.")
        groove_transformer_vae.eval()

    with torch.no_grad():
        # Run the batch loop
        metrics = batch_loop(
            dataloader_=test_dataloader,
            groove_transformer_vae=groove_transformer_vae,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            device=device,
            optimizer=None,
            kl_beta=kl_beta)

    metrics = {f"test/{key}": value for key, value in metrics.items()}
    return metrics


if __name__ == "__main__":
    # Load dataset as torch.utils.data.Dataset
    from data import MonotonicGrooveDataset

    # load dataset as torch.utils.data.Dataset
    training_dataset = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="train",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=False,
        load_as_tensor=True,
        sort_by_metadata_key="loop_id",
        hit_loss_balancing_beta=0.99,
        genre_loss_balancing_beta=0.99)

    # Balancing losses based on the imbalance in hit counts at
    #       any given time step and at a specific voice
    # ----------------------------------------------------------------------------------------------------
    # Beta value
    hit_balancing_beta = 0.99

    # get the effective number of hits per step and voice
    hits = training_dataset.outputs[:, :, :training_dataset.outputs.shape[-1]//3]
    total_hits = hits.sum(dim=0)
    effective_num_hits = 1.0 - np.power(hit_balancing_beta, total_hits)
    hit_balancing_weights = (1.0 - hit_balancing_beta) / effective_num_hits
    # normalize
    num_classes = hit_balancing_weights.shape[0] * hit_balancing_weights.shape[1]
    hit_balancing_weights = hit_balancing_weights / hit_balancing_weights.sum() * num_classes
    hit_balancing_weights_per_sample = [hit_balancing_weights for _ in range(len(training_dataset.outputs))]

    # Balancing losses based on the imbalance in styles
    # ----------------------------------------------------------------------------------------------------
    genre_balancing_beta = 0.99

    # get the effective number of genres
    genres_per_sample = [sample.metadata["style_primary"] for sample in training_dataset.hvo_sequences]

    genre_counts = {genre: genres_per_sample.count(genre) for genre in set(genres_per_sample)}

    effective_num_genres = 1.0 - np.power(genre_balancing_beta, list(genre_counts.values()))
    genre_balancing_weights = (1.0 - genre_balancing_beta) / effective_num_genres

    # normalize
    genre_balancing_weights = genre_balancing_weights / genre_balancing_weights.sum() * len(genre_counts)
    genre_balancing_weights = {genre: weight for genre, weight in zip(genre_counts.keys(), genre_balancing_weights)}

    t_steps = training_dataset.outputs.shape[1]
    n_voices = training_dataset.outputs.shape[2] // 3
    temp_row = np.ones((t_steps, n_voices))
    genre_balancing_weights_per_sample = np.array(
        [temp_row*genre_balancing_weights[sample.metadata["style_primary"]]
         for sample in training_dataset.hvo_sequences])
