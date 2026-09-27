"""SegFormer-B2 with explicit ImageNet normalization and unchanged class IDs."""
import torch
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class Segmenter(nn.Module):
    def __init__(self, source, config_only=False):
        super().__init__()
        config = SegformerConfig.from_pretrained(source)
        if list(config.depths) != [3, 4, 6, 3] or list(config.hidden_sizes) != [64, 128, 320, 512]:
            raise ValueError('Expected SegFormer-B2 architecture; refusing another model size')
        self.net = (SegformerForSemanticSegmentation(config) if config_only else
                    SegformerForSemanticSegmentation.from_pretrained(source))
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, x):
        return self.net(pixel_values=(x-self.mean)/self.std).logits
