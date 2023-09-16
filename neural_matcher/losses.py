

from torch import nn



class KeypointLoss(nn.Module):

    def __init__(self):
        super(KeypointLoss, self).__init__()

    def forward(self, n_match_res, gt):

        k = 0

