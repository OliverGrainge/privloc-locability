from models.conditionalvae.arch import build_encoder
import torch


def main():
    encoder, _ = build_encoder(encoder_name="resnet18", img_size=224, img_channels=3, pretrained=True)
    # Run a dummy forward pass to ensure weights download
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        encoder(dummy)
    return True


if __name__ == "__main__":
    main()

