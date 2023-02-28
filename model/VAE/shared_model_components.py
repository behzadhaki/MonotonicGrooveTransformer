#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import math


import torch
# --------------------------------------------------------------------------------
# ------------             Output Utils                      ---------------------
# --------------------------------------------------------------------------------


def get_hits_activation(_h, use_thres=True, thres=0.5, use_pd=False):
    _h = torch.sigmoid(_h)

    if use_thres:
        h = torch.where(_h > thres, 1, 0)

    if use_pd:
        pd = torch.rand(_h.shape[0], _h.shape[1])
        h = torch.where(_h > pd, 1, 0)

    return h


# --------------------------------------------------------------------------------
# ------------       Positinal Encoding BLOCK                ---------------------
# --------------------------------------------------------------------------------
class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)  # Shape (max_len)
        position = position.unsqueeze(1)  # Shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Insert a new dimension for batch size
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# --------------------------------------------------------------------------------
# ------------                  ENCODER BLOCK                ---------------------
# --------------------------------------------------------------------------------
class Encoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_encoder_layers, dropout, ):
        """Transformer Encoder Layers Wrapped into a Single Module"""
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.Encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=norm_encoder)

    def forward(self, src):
        """
        input and output both have shape (batch, seq_len, embed_dim)
        :param src:
        :return:
        """
        out = self.Encoder(src)
        return out


# --------------------------------------------------------------------------------
# ------------                     I/O Layers                ---------------------
# --------------------------------------------------------------------------------
class InputLayer(torch.nn.Module):
    """ Maps the dimension of the input to the dimension of the model

    eg. from (batch, 32, 27) to (batch, 32, 128)
    """
    def __init__(self, embedding_size, d_model, dropout, max_len):
        super(InputLayer, self).__init__()

        self.Linear = torch.nn.Linear(embedding_size, d_model, bias=True)
        self.ReLU = torch.nn.ReLU()
        self.PositionalEncoding = PositionalEncoding(d_model, max_len, dropout)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        x = self.Linear(src)
        x = self.ReLU(x)
        out = self.PositionalEncoding(x)

        return out


class OutputLayer(torch.nn.Module):
    """ Maps the dimension of the output of a decoded sequence into to the dimension of the output

        eg. from (batch, 32, 128) to (batch, 32, 27)
        """
    def __init__(self, embedding_size, d_model):
        """
        Output layer of the transformer model
        :param embedding_size: size of the embedding (output dim at each time step)
        :param d_model:     size of the model         (input dim at each time step)
        :param offset_activation:   activation function for the offset (default: tanh) (options: sigmoid, tanh)
        """
        super(OutputLayer, self).__init__()

        self.embedding_size = embedding_size
        self.Linear = torch.nn.Linear(d_model, embedding_size, bias=True)

    def init_weights(self, initrange=0.1, offset_activation='tanh'):
        if offset_activation == 'tanh':
            self.Linear.bias.data.zero_()
        else:
            print(f'Offset activation is {offset_activation}, bias is initialized to 0.5')
            self.Linear.bias.data.fill_(0.5)
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        y = self.Linear(decoder_out)
        y = torch.reshape(y, (decoder_out.shape[0], decoder_out.shape[1], 3, self.embedding_size // 3))
        h_logits = y[:, :, 0, :]
        v_logits = y[:, :, 1, :]
        o_logits = y[:, :, 2, :]

        # h_logits = _h
        # v = torch.sigmoid(_v)
        # if self.offset_activation == "tanh":
        #     o = torch.tanh(_o) * 0.5
        # else:
        #     o = torch.sigmoid(_o) - 0.5

        return h_logits, v_logits, o_logits


# --------------------------------------------------------------------------------
# ------------         VARIAIONAL REPARAMETERIZE BLOCK       ---------------------
# --------------------------------------------------------------------------------
class LatentLayer(torch.nn.Module):
    """ Latent variable reparameterization layer

   :param input: (Tensor) Input tensor to REPARAMETERIZE [B x max_len_enc x d_model_enc]
   :return: mu, log_var, z (Tensor) [B x max_len_enc x d_model_enc]
   """

    def __init__(self, max_len, d_model, latent_dim):
        super(LatentLayer, self).__init__()

        self.fc_mu = torch.nn.Linear(int(max_len*d_model), latent_dim)
        self.fc_var = torch.nn.Linear(int(max_len*d_model), latent_dim)

    def init_weights(self, initrange=0.1):
        self.fc_mu.bias.data.zero_()
        self.fc_mu.weight.data.uniform_(-initrange, initrange)
        self.fc_var.bias.data.zero_()
        self.fc_var.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """ converts the input into a latent space representation

        :param src: (Tensor) Input tensor to REPARAMETERIZE [N x max_encoder_len x d_model]
        :return:  mu , logvar, z (each with dimensions [N, latent_dim_size])
        """
        result = torch.flatten(src, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # Reparameterize
        z = self.reparametrize(mu, log_var)

        return mu, log_var, z

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return z
# --------------------------------------------------------------------------------
# ------------       RE-CONSTRUCTION DECODER INPUT             -------------------
# --------------------------------------------------------------------------------
class DecoderInput(torch.nn.Module):
    """ reshape the input tensor to fix dimensions with decoder

   :param input: (Tensor) Input tensor distribution [Nx(latent_dim)]
   :return: (Tensor) [N x max_len x d_model]
   """

    def __init__(self, max_len, latent_dim, d_model):
        super(DecoderInput, self).__init__()

        self.max_len = max_len
        self.d_model = d_model

        self.updims = torch.nn.Linear(latent_dim, int(max_len * d_model))

    def init_weights(self, initrange=0.1):
        self.updims.bias.data.zero_()
        self.updims.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):

        uptensor = self.updims(src)

        result = uptensor.view(-1, self.max_len, self.d_model)

        return result


class VAE_Decoder(torch.nn.Module):
    """
    Decoder for the VAE model
    This is a class, wrapping an original transformer encoder and an output layer
    into a single module.

    The implementation such that the activation functions are not hard-coded into the forward function.
    This allows for easy extension of the model with different activation functions. That said, the decode function
    is hard-coded to use sigmoid for hits and velocity and sigmoid OR tanh for offset, depending on the value of the
    o_activation parameter. The reasoning behind this is that, this way we can easily compare the performance of the
    model with different activation functions as well as different loss functions (i.e. similar to GrooVAE and
    MonotonicGrooveTransformer, we can use tanh for offsets and use MSE loss for training, OR ALTERNATIVELY,
     use sigmoid for offsets and train with BCE loss).

    """
    def __init__(self, latent_dim, d_model, num_decoder_layers, nhead, dim_feedforward,
                 output_max_len, output_embedding_size, dropout, o_activation):
        super(VAE_Decoder, self).__init__()

        assert o_activation in ["sigmoid", "tanh"]

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_decoder_layers =    num_decoder_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.output_max_len = output_max_len
        self.output_embedding_size = output_embedding_size
        self.dropout = dropout
        self.o_activation = o_activation

        self.DecoderInput = DecoderInput(
            max_len=self.output_max_len,
            latent_dim=self.latent_dim,
            d_model=self.d_model)

        self.Decoder = Encoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.num_decoder_layers,
            dropout=self.dropout)

        self.OutputLayer = OutputLayer(
            embedding_size=self.output_embedding_size,
            d_model=self.d_model)

    def forward(self, latent_z):
        """Converts the latent vector into hit, vel, offset logits (i.e. Values **PRIOR** to activation).

            :param latent_z: (Tensor) [N x latent_dim]
            :return: (Tensor) h_logits, v_logits, o_logits (each with dimension [N x max_len x num_voices])
        """
        pre_out = self.DecoderInput(latent_z)
        decoder_ = self.Decoder(pre_out)
        h_logits, v_logits, o_logits = self.OutputLayer(decoder_)

        return h_logits, v_logits, o_logits

    def decode(self, latent_z, threshold=0.5, use_thres=True, use_pd=False, return_concatenated=False):
        """Converts the latent vector into hit, vel, offset values

        :param latent_z: (Tensor) [N x latent_dim]
        :param threshold: (float) Threshold for hit prediction
        :param use_thres: (bool) Whether to use thresholding for hit prediction
        :param use_pd: (bool) Whether to use a pd for hit prediction
        :param return_concatenated: (bool) Whether to return the concatenated tensor or the individual tensors
        **For now only thresholding is supported**

        :return: (Tensor) h, v, o (each with dimension [N x max_len x num_voices])"""

        self.eval()
        with torch.no_grad():
            h_logits, v_logits, o_logits = self.forward(latent_z)
            h = get_hits_activation(h_logits, use_thres=use_thres, thres=threshold, use_pd=use_pd)
            v = torch.sigmoid(v_logits)

            if self.o_activation == "tanh":
                o = torch.tanh(o_logits) * 0.5
            elif self.o_activation == "sigmoid":
                o = torch.sigmoid(o_logits) - 0.5
            else:
                raise ValueError(f"{self.o_activation} for offsets is not supported")

        return h, v, o if not return_concatenated else torch.cat([h, v, o], dim=-1)

    def sample(self, latent_z, voice_thresholds, voice_max_count_allowed,
               return_concatenated=False, sampling_mode=0):
        """Converts the latent vector into hit, vel, offset values

        :param latent_z: (Tensor) [N x latent_dim]
        :param voice_thresholds: (list) Thresholds for hit prediction
        :param voice_max_count_allowed: (list) Maximum number of hits to allow for each voice
        :param return_concatenated: (bool) Whether to return the concatenated tensor or the individual tensors
        :param sampling_mode: (int) 0 for top-k sampling,
                                    1 for bernoulli sampling
        """
        self.eval()
        with torch.no_grad():
            h_logits, v_logits, o_logits = self.forward(latent_z)
            _h = torch.sigmoid(h_logits)
            h = torch.zeros_like(_h)

            v = torch.sigmoid(v_logits)

            if self.o_activation == "tanh":
                o = torch.tanh(o_logits) * 0.5
            elif self.o_activation == "sigmoid":
                o = torch.sigmoid(o_logits) - 0.5
            else:
                raise ValueError(f"{self.o_activation} for offsets is not supported")

            if sampling_mode == 0:
                for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                    max_indices = torch.topk(_h[:, :, ix], max_count).indices[0]
                    h[:, max_indices, ix] = _h[:, max_indices, ix]
                    h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)
            elif sampling_mode == 1:
                for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                    # sample using probability distribution of hits (_h)
                    voice_probs = _h[:, :, ix]
                    sampled_indices = torch.bernoulli(voice_probs)
                    max_indices = torch.topk(sampled_indices*voice_probs, max_count).indices[0]
                    h[:, max_indices, ix] = 1

            # sample using probability distribution of velocities (v)
            if return_concatenated:
                return torch.concat((h, v, o), -1)
            else:
                return h, v, o
