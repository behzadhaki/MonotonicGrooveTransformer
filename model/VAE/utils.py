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


