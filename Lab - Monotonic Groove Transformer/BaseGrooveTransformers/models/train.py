import os
import torch
import wandb
import re
import numpy as np
from models.transformer import GrooveTransformerEncoder, GrooveTransformer


def calculate_loss(prediction, y, bce_fn, mse_fn, hit_loss_penalty):
    y_h, y_v, y_o = torch.split(y, int(y.shape[2] / 3), 2)  # split in voices
    pred_h, pred_v, pred_o = prediction

    hit_loss_penalty_mat = torch.where(y_h == 1, float(1), float(hit_loss_penalty))

    bce_h = bce_fn(pred_h, y_h) * hit_loss_penalty_mat  # batch, time steps, voices
    bce_h_sum_voices = torch.sum(bce_h, dim=2)  # batch, time_steps
    bce_hits = bce_h_sum_voices.mean()

    mse_v = mse_fn(pred_v, y_v) * hit_loss_penalty_mat  # batch, time steps, voices
    mse_v_sum_voices = torch.sum(mse_v, dim=2)  # batch, time_steps
    mse_velocities = mse_v_sum_voices.mean()

    mse_o = mse_fn(pred_o, y_o) * hit_loss_penalty_mat
    mse_o_sum_voices = torch.sum(mse_o, dim=2)
    mse_offsets = mse_o_sum_voices.mean()

    total_loss = bce_hits + mse_velocities + mse_offsets

    _h = torch.sigmoid(pred_h)
    h = torch.where(_h > 0.5, 1, 0)  # batch=64, timesteps=32, n_voices=9

    h_flat = torch.reshape(h, (h.shape[0], -1))
    y_h_flat = torch.reshape(y_h, (y_h.shape[0], -1))
    n_hits = h_flat.shape[-1]
    hit_accuracy = (torch.eq(h_flat, y_h_flat).sum(axis=-1) / n_hits).mean()

    hit_perplexity = torch.exp(bce_hits)

    return total_loss, hit_accuracy.item(), hit_perplexity.item(), bce_hits.item(), mse_velocities.item(), \
           mse_offsets.item()


def initialize_model(params):
    model_params = params["model"]
    training_params = params["training"]
    load_model = params["load_model"]

    if model_params['encoder_only']:
        groove_transformer = GrooveTransformerEncoder(model_params['d_model'], model_params['embedding_size_src'],
                                                      model_params['embedding_size_tgt'], model_params['n_heads'],
                                                      model_params['dim_feedforward'], model_params['dropout'],
                                                      model_params['num_encoder_layers'],
                                                      model_params['max_len'], model_params['device'])
    else:
        groove_transformer = GrooveTransformer(model_params['d_model'],
                                               model_params['embedding_size_src'],
                                               model_params['embedding_size_tgt'], model_params['n_heads'],
                                               model_params['dim_feedforward'], model_params['dropout'],
                                               model_params['num_encoder_layers'],
                                               model_params['num_decoder_layers'],
                                               model_params['max_len'], model_params['device'])

    groove_transformer.to(model_params['device'])
    optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=training_params['learning_rate']) if \
        model_params['optimizer'] == 'adam' else torch.optim.SGD(groove_transformer.parameters(),
                                                                 lr=training_params['learning_rate'])
    epoch = 0

    if load_model is not None:

        # If model was saved locally
        if load_model['location'] == 'local':

            last_checkpoint = 0
            # From the file pattern, get the file extension of the saved model (in case there are other files in dir)
            file_extension_pattern = re.compile(r'\w+')
            file_ext = file_extension_pattern.findall(load_model['file_pattern'])[-1]

            # Search for all continuous digits in the file name
            ckpt_pattern = re.compile(r'\d+')
            ckpt_filename = ""

            # Iterate through files in directory, find last checkpoint
            for root, dirs, files in os.walk(load_model['dir']):
                for name in files:
                    if name.endswith(file_ext):
                        checkpoint_epoch = int(ckpt_pattern.findall(name)[-1])
                        if checkpoint_epoch > last_checkpoint:
                            last_checkpoint = checkpoint_epoch
                            ckpt_filename = name

            # Load latest checkpoint found
            if last_checkpoint > 0:
                path = os.path.join(load_model['dir'], ckpt_filename)
                checkpoint = torch.load(path)

        # If restoring from wandb
        elif load_model['location'] == 'wandb':
            model_file = wandb.restore(load_model['file_pattern'].format(load_model['run'],
                                                                         load_model['epoch']),
                                       run_path=load_model['dir'])
            checkpoint = torch.load(model_file.name)

        groove_transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    return groove_transformer, optimizer, epoch


def train_loop(dataloader, groove_transformer, loss_fn, bce_fn, mse_fn, opt, epoch, save, device,
               encoder_only, hit_loss_penalty=1, test_inputs=None, test_gt=None, validation_inputs=None,
               validation_gt=None):
    size = len(dataloader.dataset)
    groove_transformer.train()  # train mode
    loss = 0

    for batch, (x, y, idx) in enumerate(dataloader):

        opt.zero_grad()

        x = x.to(device)
        y = y.to(device)

        # Compute prediction and loss
        if encoder_only:
            pred = groove_transformer(x)
        else:
            # y_shifted
            y_s = torch.zeros([y.shape[0], 1, y.shape[2]]).to(device)
            y_s = torch.cat((y_s, y[:, :-1, :]), dim=1).to(device)
            pred = groove_transformer(x, y_s)

        loss, training_accuracy, training_perplexity, bce_h, mse_v, mse_o = loss_fn(pred, y, bce_fn, mse_fn,
                                                                                    hit_loss_penalty)

        # Backpropagation
        loss.backward()

        # update optimizer
        opt.step()

        if batch % 1 == 0:
            wandb.log({'train_loss': loss.item(), 'train_hit_accuracy': training_accuracy, 'train_hit_perplexity':
                training_perplexity, 'train_hit_loss': bce_h, 'train_velocity_loss': mse_v, 'train_offset_loss':
                mse_o, 'epoch': epoch, 'batch': batch}, commit=True)
        if batch % 100 == 0:
            print('=======')
            current = batch * len(x)
            print(f"loss: {loss.item():>4f}  [{current:>4d}/{size:>4d}]")
            print("hit accuracy:", np.round(training_accuracy, 4))
            print("hit perplexity: ", np.round(training_perplexity, 4))
            print("hit bce: ", np.round(bce_h, 4))
            print("velocity mse: ", np.round(mse_v, 4))
            print("offset mse: ", np.round(mse_o, 4))

    if save:
        save_filename = os.path.join(wandb.run.dir, "transformer_run_{}_Epoch_{}.Model".format(wandb.run.id, epoch))
        torch.save({'epoch': epoch, 'model_state_dict': groove_transformer.state_dict(),
                    'optimizer_state_dict': opt.state_dict(), 'loss': loss.item()}, save_filename)

        # save model during training (if the training crashes, models will still be available at wandb.ai)
        wandb.save(save_filename, base_path=wandb.run.dir)

    if test_inputs is not None and test_gt is not None:
        test_inputs = test_inputs.to(device)
        test_gt = test_gt.to(device)
        groove_transformer.eval()
        with torch.no_grad():
            if encoder_only:
                test_predictions = groove_transformer(test_inputs)
            else:
                # test_gt_shifted
                test_gt_s = torch.zeros([test_gt.shape[0], 1, test_gt.shape[2]]).to(device)
                test_gt_s = torch.cat((test_gt_s, test_gt[:, :-1, :]), dim=1).to(device)
                test_predictions = groove_transformer(test_inputs, test_gt_s)
            test_loss, test_hits_accuracy, test_hits_perplexity, test_bce_h, test_mse_v, test_mse_o = \
                loss_fn(test_predictions, test_gt, bce_fn, mse_fn, hit_loss_penalty)
            wandb.log({'test_loss': test_loss.item(), 'test_hit_accuracy': test_hits_accuracy,
                       'test_hit_perplexity': test_hits_perplexity, 'test_hit_loss': test_bce_h,
                       'test_velocity_loss': test_mse_v, 'test_offset_loss': test_mse_o, 'epoch': epoch},
                      commit=True)

    if validation_inputs is not None and validation_gt is not None:
        validation_inputs = validation_inputs.to(device)
        validation_gt = validation_gt.to(device)
        groove_transformer.eval()
        with torch.no_grad():
            if encoder_only:
                validation_predictions = groove_transformer(validation_inputs)
            else:
                # validation_gt_shifted
                validation_gt_s = torch.zeros([validation_gt.shape[0], 1, validation_gt.shape[2]]).to(device)
                validation_gt_s = torch.cat((validation_gt_s, validation_gt[:, :-1, :]), dim=1).to(device)
                validation_predictions = groove_transformer(validation_inputs, validation_gt_s)
            validation_loss, validation_hits_accuracy, validation_hits_perplexity, validation_bce_h, validation_mse_v, \
            validation_mse_o = loss_fn(validation_predictions, validation_gt, bce_fn, mse_fn, hit_loss_penalty)
            wandb.log({'validation_loss': validation_loss.item(), 'validation_hit_accuracy': validation_hits_accuracy,
                       'validation_hit_perplexity': validation_hits_perplexity, 'validation_hit_loss': validation_bce_h,
                       'validation_velocity_loss': validation_mse_v, 'validation_offset_loss': validation_mse_o,
                       'epoch': epoch},
                      commit=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return loss.item()
