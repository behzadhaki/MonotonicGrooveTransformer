import torch


class Encoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_encoder_layers):
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm_encoder)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # 32xNxd_model
        out = self.Encoder(src)  # 32xNxd_model
        out = out.permute(1, 0, 2)  # Nx32xd_model
        return out
