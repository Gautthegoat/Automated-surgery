import av
from datetime import timedelta, datetime, timezone
import json
import pandas as pd

def decode_s12m_timecode():
    def bcd_to_dec(bcd):
        low_nibble = bcd & 0x0F
        high_nibble = (bcd >> 4) & 0x0F
        return low_nibble + 10*high_nibble

    def decode_s12m_timecode(raw_data):
        # Extract timecode components
        h = bcd_to_dec(raw_data[4] & 0x7F)
        m = bcd_to_dec(raw_data[5] & 0x7F)
        s = bcd_to_dec(raw_data[6] & 0x7F)
        ms = bcd_to_dec(raw_data[7])
        return h, m, s, ms
    
    video_path = '/data/gb/automated_surgery_dataset/videos/2024-08-01-21-28-16.h264'
    container = av.open(video_path)
    i=0
    for frame in container.decode(video=0):
        for side_data in frame.side_data:
            if side_data.type == "S12M_TIMECODE":
                raw_data = bytes(side_data)
                # print(f"Raw data: {raw_data} / Hex: {raw_data.hex()} / len: {len(raw_data)}")
                h, m, s, ms = decode_s12m_timecode(raw_data)
                print(f"Timecode: {h}:{m}:{s}.{ms}")

        i+=1
        if i>10:
            break

def read_json():
    with open('/data/gb/automated_surgery_dataset/videos/output.json', 'r') as f:
        data = json.load(f)
    print(f'Number of frames: {len(data["frames"])}')
    timecodes = []
    for frame in data['frames']:
        if 'side_data_list' in frame:
            for side_data in frame['side_data_list']:
                if side_data['side_data_type'] == 'SMPTE 12-1 timecode':
                    timecodes.append(side_data['timecodes'][0]['value'])
    df = pd.DataFrame({'timecode': timecodes})
    print(df.head())
    df.to_csv('/data/gb/automated_surgery_dataset/videos/timecodes.csv', index=False)

if __name__ == '__main__':
    decode_s12m_timecode()
    # read_json()