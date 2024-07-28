import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class SurgicalRobotDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_index = self._create_data_index()

    def _create_data_index(self):
        data_index = []
        for demo in os.listdir(self.data_dir):
            demo_dir = os.path.join(self.data_dir, demo)
            frames_dir = os.path.join(demo_dir, 'frames')
            joint_values_path = os.path.join(demo_dir, 'joint_values.csv')
            
            joint_values = pd.read_csv(joint_values_path)
            
            for i, frame in enumerate(sorted(os.listdir(frames_dir))):
                data_index.append({
                    'frame_path': os.path.join(frames_dir, frame),
                    'joint_values': joint_values.iloc[i].values
                })
        return data_index

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        item = self.data_index[idx]
        
        # Load image
        image = Image.open(item['frame_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load joint values
        joint_values = torch.tensor(item['joint_values'], dtype=torch.float32)
        
        return image, joint_values