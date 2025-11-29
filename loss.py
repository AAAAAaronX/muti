import torch
import torch.nn as nn

class ImprovedUncertaintyLoss(nn.Module):

    def __init__(self):
        super(ImprovedUncertaintyLoss, self).__init__()
     
    def forward(self, predictions, targets, uncertainty):
        # loss = 1/(2*σ²) * (y_pred - y_true)² + log(σ)
        precision = 1.0 / (2.0 * uncertainty ** 2 + 1e-6) #
        mse_loss = precision * (predictions - targets) ** 2
        uncertainty_loss = torch.log(uncertainty + 1e-6)
     
        loss = uncertainty_loss + 0.5*mse_loss
        return torch.mean(loss)