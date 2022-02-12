from __future__ import print_function
from model.Melody_VAE import VAE
from data.data import MelodyDataset, note_to_index_dict
import argparse
import torch
from torchvision.utils import save_image
import os

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser(description='VAE MELODY Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many epochs to wait before saving trained model')
parser.add_argument('--mode', type=str, default="client", metavar='S',
                    help='')
parser.add_argument('--port', type=int, default=65522, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--measure-duration', type=int, default=16, metavar='N',
                    help='Number of time-steps (in 16th note) in a measure')
parser.add_argument('--n-measures-per-segment', type=int, default=1, metavar='N',
                    help='Number of measures in each melody')
parser.add_argument('--hop-size-in-measure', type=int, default=1, metavar='N',
                    help='Number of measures between consecutive samples')
parser.add_argument('--train-csv-directory', type=str, default="data/dataset/csv_train/*.csv", metavar='S',
                    help='Location for Training CSV files')
parser.add_argument('--test-csv-directory', type=str, default="data/dataset/csv_test/*.csv", metavar='S',
                    help='Location for Testing CSV files')

args = parser.parse_args()

torch.manual_seed(args.seed)


def most_probable_class(logits, measure_duration, n_classes_per_step):
    logits = logits.reshape(logits.shape[0], measure_duration, n_classes_per_step)
    most_probable = torch.argmax(logits, dim=2)
    predictions = torch.nn.functional.one_hot(most_probable, n_classes_per_step)
    predictions = predictions.type(dtype=torch.float32)
    return predictions.flatten(1)


def save_tensors_to_image(list_of_tensors, save_folder, measure_duration, epoch, upscale_image_by=10):

    # Create an instance of the upsample method --> Used for enlarging images by upscale_image_by
    enlarge = torch.nn.modules.upsampling.Upsample(scale_factor=upscale_image_by, mode='nearest')

    # append image rows with gaps
    image_rows = []

    # calculate the number of classes
    n_classes_per_step = int(list_of_tensors[0].shape[-1]/measure_duration)

    print("")
    for ix, tensor in enumerate(list_of_tensors):
        image_rows.append(tensor.view(tensor.shape[0], 1, measure_duration, n_classes_per_step))
        if ix < (len(list_of_tensors)-1):            # add empty space, except after last row
            image_rows.append(torch.zeros(tensor.shape[0], 1, measure_duration, n_classes_per_step))

    # convert rows in image_rows into a single tensor
    image = torch.cat(image_rows)
    image = enlarge(image)

    # save the image
    save_image(image, os.path.join(save_folder, str(epoch) + '_top_real_bottom_prediction.png'))


def save_model(mode, epoch):
    ###################
    # YOUR CODE HERE
    # Append epoch number to file name
    ###################
    pass

if __name__ == "__main__":

    # Specify the
    measure_duration = args.measure_duration                    # multiple of 16th note in 4/4 meter
    n_measures_per_segment = args.n_measures_per_segment        # number of measures for each melody
    hop_size_in_measure = args.hop_size_in_measure              # number of measures between consecutive samples

    # Training Data
    train_csv_directory = args.train_csv_directory
    train_dataset = MelodyDataset(train_csv_directory, note_to_index_dict, measure_duration, n_measures_per_segment,
                                  hop_size_in_measure)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Test Data
    test_csv_directory = args.test_csv_directory
    test_dataset = MelodyDataset(test_csv_directory, note_to_index_dict,measure_duration, n_measures_per_segment,
                                 hop_size_in_measure)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Figure out the number of unique values in the note_to_index_dict
    classes = []
    for key in note_to_index_dict.keys():
        classes.append(note_to_index_dict[key])
    n_classes_in_dataset = len(set(classes))

    # Create an instance of the model
    vae_model = VAE()

    # Create an instance of the Adam optimizer --> used for updating the weights in the model
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

    # Specify the number of random examples from the test set to test the model with
    num_test_count = 8

    # Start the training process
    for epoch in range(1, args.epochs + 1):

        # Train the model one more time on the entirety of the training set
        vae_model.train_for_one_epoch(optimizer, train_loader, epoch)

        # Test the latest version of the trained model
        vae_model.test(test_loader)

        # Every 10 epoch, conduct two evaluations (Eval 1: Reconstruction)
        # and (Eval 2: Random Generation) and save the corresponding images
        if epoch % 10 == 0 or epoch <= 10:
            with torch.no_grad():       # used for preventing weight updates during testing/inference

                # ---------------------   EVALUATION 1: Reconstructing Inputs  -----------------------

                # Grab num_test_count examples from test set to encode and then reconstruct
                inputs, _ = test_dataset.__getitem__(
                    torch.randint(low=0, high=test_dataset.__len__(), size=(num_test_count, 1)).flatten())

                # Pass data through the model (i.e. encode/sample/decode)
                reconstructed_probabilities, _, _ = vae_model(inputs)

                # Get most probable category out of all possibilities
                reconstructed_most_probable = most_probable_class(
                    reconstructed_probabilities, measure_duration, n_classes_in_dataset)

                # Save to image
                save_tensors_to_image(
                    list_of_tensors=[inputs, reconstructed_most_probable, reconstructed_probabilities],
                    save_folder="results/reconstructed/",
                    measure_duration=measure_duration,
                    epoch=epoch,
                    upscale_image_by=10)

                # ----------------------- EVALUATION 2: Generation from Scratch -----------------------
                # Randomly sample num_test_count number of z vectors
                sample = torch.randn(num_test_count, 20)  # Remember that the latent z size is 20

                # Pass random z's through the decoder to come up with a generated output
                sample = vae_model.decode(sample)

                # Get most probable category out of all possibilities
                sample_most_probable = most_probable_class(sample, measure_duration, n_classes_in_dataset)

                # Concatenate probabilities and most probable into a single image
                save_tensors_to_image(
                    list_of_tensors=[sample, sample_most_probable],
                    save_folder="results/randomly_sampled/",
                    measure_duration=measure_duration,
                    epoch=epoch,
                    upscale_image_by=10)

        if (epoch in [1, 2, 3, 4, 5]) or (epoch % args.save_interval ==0):
            save_model(vae_model, epoch)