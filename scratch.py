from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import transforms
from models.geolocation import load_model
load_dotenv()

from data.generate.yfcc100m import YFCC100MDataset
from models.transforms import get_transform

transform = get_transform("geoclip")
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

model = load_model("geoclip")
dataset = YFCC100MDataset(transform, max_shards=1)

dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)

for batch in dataloader:
    images = batch['image']
    latitudes = batch['lat']
    longitudes = batch['lon']
    idxs = batch['idx']
    
    print(f"Batch shape: {images.shape}")
    print(f"Latitudes: {latitudes.shape}")
    print(f"Longitudes: {longitudes.shape}")
    print(f"Idxs: {idxs.shape}")
    
    lat, lon = model(images)