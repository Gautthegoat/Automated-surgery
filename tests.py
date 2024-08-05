from utils.transform import Transform
from utils.config import Config as config
from data.dataset import SurgicalRobotDataset
import torch
from torchvision.utils import save_image

# dataset = SurgicalRobotDataset(config(), Transform, start_crop=210, end_crop=900)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

import numpy as np

# Your original data
data = [
    [1, 2, 3, 4, 5],
    [3, 2, 3, 4, 5],
    [0, 2, 3, 4, 5],
    [2, 2, 3, 4, 5]
]

# Convert to numpy array and transpose
data_array = np.array(data)

# Calculate mean and std for each position
means = np.mean(data_array, axis=0)
stds = np.std(data_array, axis=0)

# Normalize the data
normalized_data = (data_array - means) / stds

# Convert back to list of lists if needed
normalized_data_list = normalized_data.tolist()

print("Normalized data:")
for row in normalized_data_list:
    print([f"{x:.2f}" for x in row])