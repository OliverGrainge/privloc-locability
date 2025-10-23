import torch 
import torch.nn as nn 
import timm
from torchvision import transforms

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


class DINOv2(nn.Module): 
    def __init__(self): 
        super().__init__()
        import timm 
        self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', 
                                         pretrained=True, num_classes=0)
        
        # Unfreeze for fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        embed_dim = self.backbone.num_features
        self.head = MLPRegressor(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        features = self.backbone(x)
        return self.head(features)

def dino_transform(): 
    return transforms.Compose([
    transforms.Resize(int(518), 
                     interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
    

class SigLip(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        
        # Create SigLip model from timm
        self.backbone = timm.create_model(
            'vit_so400m_patch14_siglip_384', 
            pretrained=True, 
            num_classes=0  # Remove classification head
        )
        
        # Unfreeze for fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        # Get embedding dimension
        self.num_features = self.backbone.num_features
        embed_dim = self.num_features
        self.head = MLPRegressor(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def siglip_transform():
    """
    SigLIP uses different normalization than ImageNet models.
    Mean/std are calculated to map [0, 255] -> [-1, 1]
    """
    return transforms.Compose([
        transforms.Resize(384, 
                         interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        # SigLIP normalization (maps to [-1, 1] range)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                            std=[0.5, 0.5, 0.5])
    ])


class MegaLoc(nn.Module): 
    def __init__(self, freeze_first_n_blocks=6): 
        super().__init__()
        
        # Load pretrained MegaLoc
        megaloc_model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        
        self.backbone = megaloc_model.backbone
        self.aggregator = megaloc_model.aggregator
        
        # Freeze first N blocks (12 blocks total for DINOv2 Base)
        for i, block in enumerate(self.backbone.model.blocks):
            if i < freeze_first_n_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Freeze patch embedding
        for param in self.backbone.model.patch_embed.parameters():
            param.requires_grad = False
        
        # Unfreeze aggregator
        for param in self.aggregator.parameters():
            param.requires_grad = True
        
        embed_dim = megaloc_model.feat_dim  # 8448
        self.head = MLPRegressor(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % 14 != 0 or w % 14 != 0:
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            x = torch.nn.functional.interpolate(
                x, size=(h, w), mode='bilinear', 
                align_corners=False, antialias=True
            )
        
        features = self.backbone(x)
        features = self.aggregator(features)
        return self.head(features)

def megaloc_transform():
    """
    MegaLoc uses standard ImageNet normalization (same as DINOv2)
    Can use flexible resolutions divisible by 14
    """
    return transforms.Compose([
        transforms.Resize(518,  # Or 378, 434, etc. (divisible by 14)
                         interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    

def load_arch_and_transform(arch_name: str, **kwargs):
    """
    Load architecture by name.
    
    Args:
        arch_name: Name of the architecture ('dinov2', 'siglip')
        **kwargs: Additional arguments for architecture initialization
        
    Returns:
        Initialized model
    """
    if arch_name.lower() == 'dinov2':
        return DINOv2(**kwargs), dino_transform()
    elif arch_name.lower() == 'siglip':
        return SigLip(**kwargs), siglip_transform()
    elif arch_name.lower() == 'megaloc':
        return MegaLoc(**kwargs), megaloc_transform()
    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: dinov2, siglip")

