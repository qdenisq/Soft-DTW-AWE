import numpy as np
import torch.nn as nn
import torch

from src.soft_dtw import SoftDTW


def anneal_function(anneal_func, step, k, x0):
    if anneal_func == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_func == 'linear':
        return min(1, step / x0)


class TripletNetLoss(nn.Module):
    """Custom loss function for the triplet network.

    It is a weighted sum of Cross-entropy loss and contrastive loss.
    Loss has a general form of l = alpha*ce_loss + (1-alpha)*contrastive_loss.
    Contrastive loss accepts different metrics to compute distance between samples.
    Metric can be on of "sdtw", "l2", "cos_hinge", "ce"
    If "ce" is chosen, then contrastive loss is not computed.
    """

    def __init__(self, params):
        """Initializes loss.

        Creates SoftDTW object if the type of the loss is set to "sdtw".

        Parameters
        ----------
        params: dict
            dictrionary of parameters
        """
        super(TripletNetLoss, self).__init__()
        self.params = params
        if params['type'] == 'sdtw':
            self.sdtw = SoftDTW(open_end=params['open_end'], dist=params['dist'])

    def forward(self, predicts, targets, embeds, step):
        """Computes loss.

        Parameters
        ----------
        predicts: torch.Tensor
            Predicted labels
        targets: torch.Tensor
            True labels
        embeds: torch.Tensor
            Embeddings of the inputs
        step: int
            Value to compute the annealing factor for triplet loss.

        Returns
        -------
        torch.Tensor
            computed loss
        """
        alpha = self.params['alpha']
        margin = self.params['margin']
        n_mini_batch_size = embeds[0].shape[0] // 2

        # ce loss
        ce_loss = torch.nn.CrossEntropyLoss()(predicts, targets)

        Triplet_loss_weight = anneal_function('logistic', step, self.params['triplet_anneal_k'],
                                              self.params['triplet_anneal_b'])

        if self.params['type'] == 'sdtw':
            # DTWLoss (want to minimize dtw between duplicates and maximize dtw between non-duplicates)
            DTW_loss = torch.tensor([0]).float().to(embeds[0].device)
            for k in range(n_mini_batch_size):
                DTW_loss += torch.nn.functional.relu(self.sdtw(embeds[0][k], embeds[1][k]) -
                                                     self.sdtw(embeds[0][k + n_mini_batch_size],
                                                                   embeds[1][k + n_mini_batch_size])
                                                     + margin)

            DTW_loss /= (n_mini_batch_size)

            Triplet_loss = DTW_loss
            loss = alpha * ce_loss + (1. - alpha) * Triplet_loss * Triplet_loss_weight

        elif self.params['loss_type'] == 'l2':
            L2_loss = torch.nn.functional.relu(
                torch.sum((embeds[0][:n_mini_batch_size, -1, :] - embeds[1][:n_mini_batch_size, -1, :]) ** 2, dim=-1) -
                torch.sum((embeds[0][:n_mini_batch_size, -1, :] - embeds[1][n_mini_batch_size:, -1, :]) ** 2, dim=-1) +
                margin).sum()
            L2_loss /= n_mini_batch_size
            loss = alpha * ce_loss + (1. - alpha) * L2_loss * Triplet_loss_weight

        elif self.params['loss_type'] == 'cos_hinge':
            Cos_hinge_loss = torch.clamp_min(
                - torch.nn.CosineSimilarity(dim=-1)(embeds[0][:n_mini_batch_size, -1, :],
                                                    embeds[1][:n_mini_batch_size, -1, :])
                + torch.nn.CosineSimilarity(dim=-1)(embeds[0][:n_mini_batch_size, -1, :],
                                                    embeds[1][n_mini_batch_size:, -1, :])
                + margin, 0).sum()

            Cos_hinge_loss += torch.clamp_min(
                - torch.nn.CosineSimilarity(dim=-1)(embeds[0][:n_mini_batch_size, -1, :],
                                                    embeds[1][:n_mini_batch_size, -1, :])
                + torch.nn.CosineSimilarity(dim=-1)(embeds[1][:n_mini_batch_size, -1, :],
                                                    embeds[1][n_mini_batch_size:, -1, :])
                + margin, 0).sum()

            Cos_hinge_loss /= (2 * n_mini_batch_size)
            loss = alpha * ce_loss + (1. - alpha) * Cos_hinge_loss * Triplet_loss_weight
        elif self.params['loss_type'] == 'ce':
            loss = ce_loss
        else:
            raise KeyError(f"Unknown loss type: {self.params['loss_type']}")

        return loss


