import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import ast

class SurgicalRobotDataset(Dataset):
    def __init__(self, config, transform, start_crop=300, end_crop=900):
        self.data_dir = config.data_dir
        self.nb_futur_logs = config.chunk_size
        self.height = config.image_size[0]
        self.width = config.image_size[1]
        self.mode = config.type_predictions
        self.type_norm = config.type_norm
        self.transform_frame = transform.transform_frame
        self.transform_log = transform.transform_log
        self.videos_path = os.path.join(self.data_dir, 'videos')
        self.frames_path = os.path.join(self.data_dir, 'frames')
        self.basic_logs = []
        self.type_logs = []
        self.create_basic_logs()
        config.stats = self.apply_norm()
        self.create_type_logs()
        self.basic_logs = self.basic_logs[start_crop:len(self.basic_logs)-end_crop]
        self.type_logs = self.type_logs[start_crop:len(self.type_logs)-end_crop]

    def apply_norm(self):
        """
        Ǹormalize the basic logs
        """
        stats = []
        data_array = np.array([logs for frame_path, logs in self.basic_logs])

        if self.type_norm == 'meanstd':
            means = np.mean(data_array, axis=0)
            stds = np.std(data_array, axis=0)
            normalized_data = (data_array - means) / stds
            print(f"Means / Joint1-6: {means[:6]}")
            print(f"Means / Joint7-12: {means[6:12]}")
            print(f"Stds / Joint1-6: {stds[:6]}")
            print(f"Stds / Joint7-12: {stds[6:12]}")
            stats = [means, stds]
        elif self.type_norm == 'minmax':
            eps = 1e-8
            mins = np.min(data_array, axis=0)
            maxs = np.max(data_array, axis=0)
            normalized_data = (data_array - mins) / ((maxs - mins)+eps)
            print(f"Mins: {mins}")
            print(f"Maxs: {maxs}")
            stats = [mins, maxs]

        normalized_data = normalized_data.tolist()

        for i, (frame_path, logs) in enumerate(self.basic_logs):
            self.basic_logs[i] = (frame_path, normalized_data[i])
            print(f'Applied normalization {self.type_norm} to {i+1}/{len(self.basic_logs)} logs', end='\r')
        print(f'Applied normalization {self.type_norm} to {i+1}/{len(self.basic_logs)} logs')
        return stats

    def apply_absolute_joint_prediction(self):
        """
        Apply absolute joint prediction to the logs, except for joint 5 and 6 for left and right since they depend on initial position
        """
        self.type_logs = self.basic_logs
        print('Applied Absolute joint prediction to all logs')

    def apply_delta_joint_prediction(self):
        """
        Apply delta joint prediction to the logs
        """
        for i, (frame_path, logs) in enumerate(self.basic_logs):
            if i == 0:
                self.type_logs.append((frame_path, logs))
                continue
            new_logs = []
            for j in range(len(logs)):
                new_logs.append(logs[j]-self.basic_logs[i-1][1][j])
            self.type_logs.append((frame_path, tuple(new_logs)))
            print(f"Applying Delta joint prediction {i}/{len(self.basic_logs)}", end='\r')
        print(f'Applied Delta joint prediction to {i+1}/{len(self.basic_logs)} frames')

    def create_basic_logs(self):
        """
        Go through all the frames and logs and create a list of tuples (frame_path, logs)
        """
        print('Indexing Dataset...')
        for video in os.listdir(self.videos_path):
            print(f'***Processing video {video}***')
            video_name = video.split('.')[0]
            data_path = os.path.join(self.frames_path, f'{video_name}/data.csv')
            csv_df = pd.read_csv(data_path)
            for i, row in csv_df.iterrows():
                frame_path = row['frame']
                logs = row['logs']
                try:
                    logs = tuple(ast.literal_eval(logs).values())
                    self.basic_logs.append((frame_path, logs))
                except:
                    print(f"Error reading frame : {frame_path}, replacing logs by previous logs")
                    self.basic_logs.append((frame_path, self.basic_logs[-1][1]))
                print(f"Extracting data {i}/{len(csv_df)}", end='\r')
            print(f"Extracted {i+1}/{len(csv_df)} frames from video {video_name}")

    def create_type_logs(self):
        """
        Go through all the basic logs and create a list of logs with the desired type of prediction
        """
        if self.mode == 'Absolute_joints':
            self.apply_absolute_joint_prediction()
        elif self.mode == 'Delta_joints':
            self.apply_delta_joint_prediction()
        elif self.mode == 'Absolute position':
            print('Mode Absolute position not implemented yet, results in default mode, Absolute_joint')
        elif self.mode == 'Delta position':
            print('Mode Delta position not implemented yet')
        else:
            print('Mode not recognized, results in default mode, Absolute_joints')
            self.apply_absolute_joint_prediction()

    def __len__(self):
        """
        Necessary to implement a Dataset
        """
        return len(self.basic_logs)-self.nb_futur_logs
    
    def __getitem__(self, idx):
        """
        frame: torch.Tensor of shape (height, width, 3)
        logs: torch.Tensor of shape (nb_logs,)
        """
        # Get current frame and logs
        frame_path, logs = self.basic_logs[idx]
        frame = np.load(frame_path) # (height, width, 3)
        frame = self.transform_frame(self.height, self.width, frame) # (3, height, width)
        logs = np.array(logs) # (nb_logs)
        logs = self.transform_log(logs) # (1, nb_logs)
        
        # Get future logs
        for i in range(1, self.nb_futur_logs+1):
            _, logs_future = self.type_logs[idx+i]
            logs_future = np.array(logs_future) # (nb_logs)
            logs_future = self.transform_log(logs_future) # (1, nb_logs)
            logs = torch.cat((logs, logs_future), dim=0) # (nb_futur_logs, nb_logs)
        return frame, logs