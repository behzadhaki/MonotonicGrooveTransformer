import torch
from BaseGrooveTransformers.models.utils import PositionalEncoding


class InputLayer(torch.nn.Module):
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
    def __init__(self, embedding_size, d_model):
        super(OutputLayer, self).__init__()

        self.embedding_size = embedding_size
        self.Linear = torch.nn.Linear(d_model, embedding_size, bias=True)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        y = self.Linear(decoder_out)
        y = torch.reshape(y, (decoder_out.shape[0], decoder_out.shape[1], 3, self.embedding_size // 3))

        _h = y[:, :, 0, :]
        _v = y[:, :, 1, :]
        _o = y[:, :, 2, :]

        h = _h
        v = torch.sigmoid(_v)
        o = torch.tanh(_o) * 0.5

        return h, v, o
