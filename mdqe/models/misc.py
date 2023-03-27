import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def make_reference_points(spatial_shape, device=None):
    """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
    H_, W_ = spatial_shape
    ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                  torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
    ref_xy = torch.stack((ref_x.reshape(-1) / max(W_, 1), ref_y.reshape(-1) / max(H_, 1)), -1)  # H_lW_lx2
    ref_xy.requires_grad = False

    return ref_xy
