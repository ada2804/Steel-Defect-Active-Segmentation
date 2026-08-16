import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        """
        Combines BCE and Dice Loss to handle extreme class imbalance.
        Args:
            bce_weight (float): The weight applied to the BCE component.
            dice_weight (float): The weight applied to the Dice component.
            smooth (float): A tiny constant to prevent division by zero.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. Calculate Standard BCE Loss (Calculates directly from logits for numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # 2. Calculate Dice Loss
        # We must apply sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(logits)
        
        # Flatten the tensors from [Batch, Channels, Height, Width] to a 1D array
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate the intersection and the denominator
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # Since we want to MINIMIZE loss, we subtract the score from 1
        dice_loss = 1.0 - dice_score
        
        # 3. Combine and return
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)