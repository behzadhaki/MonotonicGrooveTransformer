import torch

from BaseGrooveTransformers.models.encoder import Encoder
from BaseGrooveTransformers.models.decoder import Decoder
from BaseGrooveTransformers.models.io_layers import InputLayer, OutputLayer
from BaseGrooveTransformers.models.utils import get_tgt_mask, get_hits_activation


class GrooveTransformer(torch.nn.Module):
    def __init__(self, d_model, embedding_size_src, embedding_size_tgt, nhead, dim_feedforward, dropout,
                 num_encoder_layers, num_decoder_layers, max_len, device):
        super(GrooveTransformer, self).__init__()

        self.d_model = d_model
        self.embedding_size_src = embedding_size_src
        self.embedding_size_tgt = embedding_size_tgt
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.device = device

        self.InputLayerEncoder = InputLayer(embedding_size_src, d_model, dropout, max_len)
        self.Encoder = Encoder(d_model, nhead, dim_feedforward, dropout, num_encoder_layers)

        self.InputLayerDecoder = InputLayer(embedding_size_tgt, d_model, dropout, max_len)
        self.Decoder = Decoder(d_model, nhead, dim_feedforward, dropout, num_decoder_layers)
        self.OutputLayer = OutputLayer(embedding_size_tgt, d_model)

        self.InputLayerEncoder.init_weights()
        self.OutputLayer.init_weights()

    def forward(self, src, tgt):
        # src Nx32xembedding_size_src
        # tgt Nx32xembedding_size_tgt
        mask = get_tgt_mask(self.max_len).to(self.device)

        x = self.InputLayerEncoder(src)  # Nx32xd_model
        y = self.InputLayerDecoder(tgt)  # Nx32xd_model
        memory = self.Encoder(x)  # Nx32xd_model
        out = self.Decoder(y, memory, tgt_mask=mask)  # Nx32xd_model
        out = self.OutputLayer(out)  # (Nx32xembedding_size_src/3,Nx32xembedding_size_src/3,Nx32xembedding_size_src/3)

        return out

    def predict(self, src, use_thres=True, thres=0.5, use_pd=False):
        self.eval()

        with torch.no_grad():
            n_voices = self.embedding_size_tgt // 3
            mask = get_tgt_mask(self.max_len).to(self.device)

            # encoder
            x = self.InputLayerEncoder(src)  # Nx32xd_model
            memory = self.Encoder(x)  # Nx32xd_model

            # init shifted target
            tgt_shift = torch.zeros([src.shape[0], self.max_len + 1, self.embedding_size_tgt]).to(self.device)

            for i in range(self.max_len):
                # decoder
                y_shift = self.InputLayerDecoder(tgt_shift[:, :-1, :])
                out = self.Decoder(y_shift, memory, tgt_mask=mask)  # Nx32xd_model
                _h, v, o = self.OutputLayer(out)

                h = get_hits_activation(_h, use_thres=use_thres, thres=thres, use_pd=use_pd)

                tgt_shift[:, i + 1, 0: n_voices] = h[:, i, :]
                tgt_shift[:, i + 1, n_voices: 2 * n_voices] = v[:, i, :]
                tgt_shift[:, i + 1, 2 * n_voices:] = o[:, i, :]

            # undo the shifting
            tgt = tgt_shift[:, 1:, :]

            # reshape
            out = tgt.reshape([tgt.shape[0], tgt.shape[1], 3, tgt.shape[2] // 3]).to(self.device)
            h = out[:, :, 0, :]
            v = out[:, :, 1, :]
            o = out[:, :, 2, :]

        return h, v, o


class GrooveTransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, embedding_size_src, embedding_size_tgt, nhead, dim_feedforward, dropout,
                 num_encoder_layers, max_len, device):
        super(GrooveTransformerEncoder, self).__init__()

        self.d_model = d_model
        self.embedding_size_src = embedding_size_src
        self.embedding_size_tgt = embedding_size_tgt
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.num_encoder_layers = num_encoder_layers
        self.device = device

        self.InputLayerEncoder = InputLayer(embedding_size_src, d_model, dropout, max_len)
        self.Encoder = Encoder(d_model, nhead, dim_feedforward, dropout, num_encoder_layers)
        self.OutputLayer = OutputLayer(embedding_size_tgt, d_model)

        self.InputLayerEncoder.init_weights()
        self.OutputLayer.init_weights()

    def forward(self, src):
        # src Nx32xembedding_size_src
        x = self.InputLayerEncoder(src)  # Nx32xd_model
        memory = self.Encoder(x)  # Nx32xd_model
        out = self.OutputLayer(
            memory)  # (Nx32xembedding_size_tgt/3,Nx32xembedding_size_tgt/3,Nx32xembedding_size_tgt/3)

        return out

    def predict(self, src, use_thres=True, thres=0.5, use_pd=False):
        self.eval()
        with torch.no_grad():
            _h, v, o = self.forward(
                src)  # Nx32xembedding_size_src/3,Nx32xembedding_size_src/3,Nx32xembedding_size_src/3

            h = get_hits_activation(_h, use_thres=use_thres, thres=thres, use_pd=use_pd)

        return h, v, o
