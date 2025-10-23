import torch 
import torch.nn as nn 
import timm
from torchvision import transforms
from typing import Optional, List

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x):
        out = self.net(x).squeeze(-1)
        return torch.nn.functional.softplus(out)  # Ensures log(1+e) >= 0


class MultiKHead(nn.Module):
    """
    Binary multi-K head: outputs m logits, one per distance threshold K.
    Use with BCEWithLogitsLoss. At inference, apply torch.sigmoid.
    """
    def __init__(self, in_dim: int, K_km: List[float],
                 hidden: int = 768, dropout: float = 0.1):
        super().__init__()
        self.K = torch.tensor(K_km, dtype=torch.float32)  # not registered as buffer on purpose
        self.m = len(K_km)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.m)  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, m] (raw logits)

class DINOv2(nn.Module):
    def __init__(self, head_type: str = 'regressor',
                 K_km: Optional[List[float]] = None,
                 head_hidden: int = 768, head_dropout: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True, num_classes=0
        )
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.feat_dim = self.backbone.num_features
        self.head_type = head_type.lower()
        if self.head_type == 'multik':
            assert K_km is not None and len(K_km) > 0, "Provide K_km for Multi-K head."
            self.head = MultiKHead(self.feat_dim, K_km, hidden=head_hidden, dropout=head_dropout)
        else:
            self.head = MLPRegressor(self.feat_dim, hidden=head_hidden, dropout=head_dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encode(x)
        return self.head(feats)


def dino_transform():
    return transforms.Compose([
        transforms.Resize(int(518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class SigLip(nn.Module):
    def __init__(self, head_type: str = 'regressor',
                 K_km: Optional[List[float]] = None,
                 head_hidden: int = 768, head_dropout: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_so400m_patch14_siglip_384',
            pretrained=True, num_classes=0
        )
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.feat_dim = self.backbone.num_features
        self.head_type = head_type.lower()
        if self.head_type == 'multik':
            assert K_km is not None and len(K_km) > 0, "Provide K_km for Multi-K head."
            self.head = MultiKHead(self.feat_dim, K_km, hidden=head_hidden, dropout=head_dropout)
        else:
            self.head = MLPRegressor(self.feat_dim, hidden=head_hidden, dropout=head_dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encode(x)
        return self.head(feats)


def siglip_transform():
    """
    SigLIP uses different normalization than ImageNet models.
    Mean/std are calculated to map [0, 255] -> [-1, 1].
    """
    return transforms.Compose([
        transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])


class MegaLoc(nn.Module):
    def __init__(self, freeze_first_n_blocks: int = 6,
                 head_type: str = 'regressor',
                 K_km: Optional[List[float]] = None,
                 head_hidden: int = 768, head_dropout: float = 0.1):
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
        self.head_type = head_type.lower()
        if self.head_type == 'multik':
            assert K_km is not None and len(K_km) > 0, "Provide K_km for Multi-K head."
            self.head = MultiKHead(self.feat_dim, K_km, hidden=head_hidden, dropout=head_dropout)
        else:
            self.head = MLPRegressor(self.feat_dim, hidden=head_hidden, dropout=head_dropout)

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


def load_arch_and_transform(arch_name: str, head_type: str, **kwargs):
    """
    Load architecture by name.

    Args:
        arch_name: 'dinov2' | 'siglip' | 'megaloc'
        **kwargs: accepts:
            - head_type: 'regressor' (default) or 'multik'
            - K_km: list[float] (required if head_type='multik')
            - head_hidden, head_dropout, freeze_first_n_blocks (for megaloc)
    Returns:
        (model, transform)
    """

    if arch_name.lower() == 'dinov2':
        return DINOv2(head_type=head_type, **kwargs), dino_transform()
    elif arch_name.lower() == 'siglip':
        return SigLip(head_type=head_type, **kwargs), siglip_transform()
    elif arch_name.lower() == 'megaloc':
        return MegaLoc(head_type=head_type, **kwargs), megaloc_transform()
    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: dinov2, siglip, megaloc")