import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """" Attention-based Deep MIL from Ilse et al: https://arxiv.org/abs/1802.04712
    """
    def __init__(self, feature_dim, hidden_dim, output_dim, drop_out=0.2):
        super(AttentionMIL, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(feature_dim, output_dim)

    def forward(self, bag, mask):
        """ Classic AttentionMIL, using masking to disregard padded instances.
        Args:
            bag: (batch_size, bag_size, feature_dim)
            mask: (batch_size, bag_size)
        Returns:
            logits: (batch_size)
            attn_weights: (batch_size, bag_size)
        """
        attn_weights = self.attention(bag).squeeze()
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))            # mask out padded instances
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_features = torch.sum(bag * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(weighted_features).squeeze()
        return logits, attn_weights
