import torch


class Decoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_decoder_layers):
        super(Decoder, self).__init__()
        norm_decoder = torch.nn.LayerNorm(d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers, norm_decoder)

    def forward(self, tgt, memory, tgt_mask):
        # tgt    Nx32xd_model
        # memory Nx32xd_model

        tgt = tgt.permute(1, 0, 2)  # 32xNxd_model
        memory = memory.permute(1, 0, 2)  # 32xNxd_model

        out = self.Decoder(tgt, memory, tgt_mask)  # 32xNxd_model

        out = out.permute(1, 0, 2)  # Nx32xd_model

        return out
