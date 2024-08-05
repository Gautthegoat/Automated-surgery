import av
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from datetime import timedelta, datetime, timezone
import os
import pandas as pd
import sqlite3

class Extract_Frames:
    """
    Extract frames and timestamps from a video file.
    """
    def __init__(self, video_path, output_folder, fps=30, verbose=False):
        self.verbose = verbose
        self.video_path = video_path
        self.output_folder = output_folder
        self.fps = fps
        self.container = av.open(video_path)
        self.df_timestamps = pd.DataFrame(columns=['timestamp', 'frame'])
        self.extract_image_timestamp()

    def bcd_to_dec(self, bcd):
        low_nibble = bcd & 0x0F
        high_nibble = (bcd >> 4) & 0x0F
        return low_nibble + 10*high_nibble

    def decode_s12m_timecode(self, side_data):
        raw_data = bytes(side_data)
        # Extract timecode components
        h = self.bcd_to_dec(raw_data[4] & 0x7F)
        m = self.bcd_to_dec(raw_data[5] & 0x7F)
        s = self.bcd_to_dec(raw_data[6] & 0x7F)
        ms = self.bcd_to_dec(raw_data[7]) / self.fps * 1000
        return h, m, s, ms
    
    def to_iso8601(self, h, m, s, ms):
        date_string = self.video_path.split('/')[-1].split('.')[0]
        date_object = datetime.strptime(date_string, "%Y-%m-%d-%H-%M-%S")
        date_object = date_object.replace(hour=h, minute=m, second=s, microsecond=int(ms)*1000, tzinfo=timezone.utc)
        return date_object.isoformat()
        
    
    def extract_timestamps(self, frame):
        for side_data in frame.side_data:
            if side_data.type == 'S12M_TIMECODE':
                return self.decode_s12m_timecode(side_data)
            else:
                return None
            
    def save_image(self, frame, timestamp, i):
        array = frame.to_ndarray(format='rgb24')
        file_path = self.output_folder + f'/frame_{i}.npy'
        np.save(file_path, array)
        new_row = pd.DataFrame({'timestamp': [timestamp], 'frame': [file_path]})
        self.df_timestamps = pd.concat([self.df_timestamps, new_row], ignore_index=True)

    def extract_image_timestamp(self):
        i = 0
        for frame in self.container.decode(video=0):
            h, m, s, ms = self.extract_timestamps(frame)
            timestamp = self.to_iso8601(h, m, s, ms)
            self.save_image(frame, timestamp, i)
            i += 1
            if self.verbose:
                print(f'Extracted frame {i}', end='\r')
        self.df_timestamps.to_csv(self.output_folder + '/data.csv', index=False)
        self.df_timestamps.to
            

class Process_Videos_Logs():
    """
    Process videos in a folder and extract frames and timestamps.
    """
    def __init__(self, data_path, fps=30, verbose=False):
        self.verbose = verbose
        self.data_path = data_path
        self.fps = fps
        self.video_path = os.path.join(data_path, 'videos')
        self.frames_path = os.path.join(data_path, 'frames')
        self.videos = [f for f in os.listdir(self.video_path) if f.endswith('.h264')]
        self.process_videos()
        self.add_logs(self.videos[0], verbose=True)

    def process_videos(self):
        for video in self.videos:
            if self.verbose:
                print(f'*****Processing video {video}')
            video_path = os.path.join(self.video_path, video)
            output_folder = os.path.join(self.frames_path, video.split('.')[0])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                if self.verbose:
                    print(f'Extracting frames for video {video}')
                Extract_Frames(video_path, output_folder, fps=self.fps, verbose=self.verbose)
                if self.verbose:
                    print(f'Adding logs for video {video}')
                self.add_logs(video, verbose=self.verbose)

            else:
                if self.verbose:
                    print(f'Frames already extracted for video {video}')

    def add_logs(self, video, verbose=False):
        # Load timestamps from frames
        timestamps_path = os.path.join(self.frames_path, video.split('.')[0], 'data.csv')
        df_timestamps = pd.read_csv(timestamps_path)
        df_timestamps['logs'] = None

        # Load logs from database
        log_path = os.path.join(self.data_path, 'logs', video.split('.')[0])
        columns = ['unix_timestamp', 
                'robot_left_j1', 'robot_left_j2', 'robot_left_j3', 
                'robot_left_j4', 'robot_left_j5', 'robot_left_j6', 
                'robot_right_j1', 'robot_right_j2', 'robot_right_j3', 
                'robot_right_j4', 'robot_right_j5', 'robot_right_j6']
        columns_str = ', '.join(columns)
        conn = sqlite3.connect(log_path + '/device_frames.db3')
        query = f"SELECT {columns_str} FROM device_frame"
        df_logs_device_frame = pd.read_sql_query(query, conn)
        conn.close()

        # Convert timestamps to datetime objects
        df_timestamps['timestamp'] = pd.to_datetime(df_timestamps['timestamp'], format='ISO8601')
        df_logs_device_frame['unix_timestamp'] = pd.to_datetime(df_logs_device_frame['unix_timestamp'], format='ISO8601')

        # Find the closest match for each timestamp
        def find_closest_log(timestamp):
            idx = np.searchsorted(df_logs_device_frame['unix_timestamp'], timestamp, side="left")
            if idx > 0 and (idx == len(df_logs_device_frame) or 
                            abs(timestamp - df_logs_device_frame['unix_timestamp'].iloc[idx-1]) < 
                            abs(timestamp - df_logs_device_frame['unix_timestamp'].iloc[idx])):
                return idx - 1
            
            else:
                return idx

        print('Finding closest logs...')
        closest_indices = df_timestamps['timestamp'].apply(find_closest_log)

        # Add the joint values to the logs column
        df_timestamps['logs'] = closest_indices.apply(lambda idx: df_logs_device_frame.iloc[idx][columns[1:]].to_dict())
        # # Save the updated DataFrame
        df_timestamps.to_csv(timestamps_path, index=False)

        if self.verbose:
            print(f"Logs added to {timestamps_path}")

if __name__ == '__main__':
    data_path = '/data/gb/automated_surgery_dataset/'
    Process_Videos_Logs(data_path, verbose=True)
