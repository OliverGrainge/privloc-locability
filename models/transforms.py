from torchvision import transforms
from typing import Optional

def get_pretrained_transform(model_name: str): 
    model_name = model_name.lower()
    if model_name == "geoclip":
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    else:
        raise ValueError(f"Model {model_name} not found")


def get_transform(model_name: Optional[str] = None, mode: str = "train"):
    model_name = model_name.lower()
    if model_name is not None:
        return get_pretrained_transform(model_name)
    else:
        raise NotImplementedError("No model name provided")