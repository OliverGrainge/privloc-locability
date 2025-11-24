from plonk.pipe import PlonkPipeline

pipeline = PlonkPipeline("nicolas-dufour/PLONK_YFCC")

for param in pipeline.parameters():
    param.requires_grad = False

from PIL import Image
import numpy as np

# Create a random noise image (e.g., 224x224 RGB)
noise = np.random.rand(224, 224, 3) * 255
noise = noise.astype(np.uint8)
img = Image.fromarray(noise, 'RGB')
images = [img]

gps_coords = pipeline.compute_likelihood(images, coordinates=[[20.0, 20.0]])
print(gps_coords)