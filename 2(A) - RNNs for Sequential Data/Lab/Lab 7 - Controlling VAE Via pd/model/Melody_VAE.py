import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(224, 100)
        self.fc21 = nn.Linear(100, 20)
        self.fc22 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Linear(100, 224)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train_for_one_epoch(self, optimizer, train_loader, epoch, log_interval=10):
        # Activate the training state of the model
        self.train()

        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.forward(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

    def test(self, test_loader):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data
                recon_batch, mu, logvar = self.forward(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def generate_random_sequence(self, min_num_notes=1):
        # min_num_notes ==> minimum number of notes in the sequence

        starts = []
        notes = []
        durations = []

        while starts == [] or len(starts) < min_num_notes:

            # Randomly sample num_test_count number of z vectors
            sample_z = torch.randn(1, 20)  # Remember that the latent z size is 20

            # Pass random z's through the decoder to come up with a generated output
            sample_logits = self.decode(sample_z)

            # convert logits to most probable sequence and individual lists of starts, notes and durations
            starts, notes, durations, sample_most_probable = self.logits_to_starts_notes_durations(sample_logits)

        return starts, notes, durations, sample_most_probable, sample_z

    def logits_to_starts_notes_durations(self, logits):

        starts = []
        notes = []
        durations = []

        # Get most probable category out of all possibilities
        sample_most_probable = self.most_probable_class(logits, measure_duration=16, n_classes_per_step=14)
        sample_most_probable = torch.argmax(sample_most_probable.view(16, 14), dim=1)
        # start_time = datetime.now() + \
        #             timedelta(seconds=self.generation_configs["grace_time_before_playback"])

        for time_step, event_in_time_step in enumerate(sample_most_probable):
            if event_in_time_step > 1:  # if event is a pitch (i.e. any class except silence and hold)
                starts.append(time_step)
                notes.append(int(event_in_time_step))

                # check for how many steps the note should be held
                hold_count = 0
                for next_step in range(time_step + 1, sample_most_probable.shape[0]):
                    if sample_most_probable[next_step] != 0:
                        break
                    else:
                        hold_count += 1

                durations.append(hold_count + 1)

        return starts, notes, durations, sample_most_probable

    def most_probable_class(self, logits, measure_duration, n_classes_per_step):
        logits = logits.reshape(logits.shape[0], measure_duration, n_classes_per_step)
        most_probable = torch.argmax(logits, dim=2)
        predictions = torch.nn.functional.one_hot(most_probable, n_classes_per_step)
        predictions = predictions.type(dtype=torch.float32)
        return predictions.flatten(1)


vae = VAE()      # Create an instance of the vae model
vae.load_state_dict(torch.load("checkpoints/epoch_1000.pt"))    # load pretrained parameters
vae.eval()      # Set to eval to signal the model that we are just using the model for inference
vae.generate_random_sequence()