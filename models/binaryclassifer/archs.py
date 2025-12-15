import torch 
import torch.nn as nn 
import timm
from torchvision import transforms

class MLPClassifier(nn.Module):
    """
    Simple linear classifier head.
    Outputs a single logit per example (use with BCEWithLogitsLoss).
    At inference, apply torch.sigmoid to output probability.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        out = self.linear(x).squeeze(-1)  # [B,]
        return out  # Raw logits for binary classification





class MegaLoc(nn.Module):
    def __init__(self, freeze_first_n_blocks: int = 6):
        super().__init__()

        megaloc_model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        self.backbone = megaloc_model.backbone
        self.aggregator = megaloc_model.aggregator

        # Freeze first N transformer blocks
        for i, block in enumerate(self.backbone.model.blocks):
            req = (i >= freeze_first_n_blocks)
            for p in block.parameters():
                p.requires_grad = req

        # Freeze patch embed
        for p in self.backbone.model.patch_embed.parameters():
            p.requires_grad = False

        # Unfreeze aggregator
        for p in self.aggregator.parameters():
            p.requires_grad = True

        self.feat_dim = megaloc_model.feat_dim  # 8448
        self.head = MLPClassifier(self.feat_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % 14 != 0 or w % 14 != 0:
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            x = torch.nn.functional.interpolate(
                x, size=(h, w), mode='bilinear',
                align_corners=False, antialias=True
            )
        feats = self.backbone(x)
        feats = self.aggregator(feats)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encode(x)
        return self.head(feats)


def megaloc_transform():
    """
    MegaLoc uses standard ImageNet normalization (same as DINOv2).
    Resolution should be divisible by 14.
    """
    return transforms.Compose([
        transforms.Resize(378, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(378),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def load_arch_and_transform(arch_name: str, **kwargs):
    """
    Load architecture by name.

    Args:
        arch_name: 'dinov2' | 'siglip' | 'megaloc'
        **kwargs: accepts:
            - freeze_first_n_blocks (for megaloc only)
    Returns:
        (model, transform)
    """

    if arch_name.lower() == 'megaloc':
        return MegaLoc(**kwargs), megaloc_transform()
    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: dinov2, siglip, megaloc")