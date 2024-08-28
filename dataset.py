from torch.utils.data import Dataset
import os
import numpy as np
import torchaudio
import decord
from decord import VideoReader, cpu

torchaudio.set_audio_backend("soundfile")
decord.bridge.set_bridge('torch')


class LLP(Dataset):
    def __init__(self, video_dir_path, audio_dir_path,
                 video_file_extension = ".mp4", 
                 audio_file_extension = ".wav",
                 subset = None) -> None:
        
        self.video_dir_path = video_dir_path
        self.audio_dir_path = audio_dir_path
        self.video_file_extension = video_file_extension
        self.audio_file_extension = audio_file_extension
        self.videos_ids = [video_id.replace(".mp4", "") for video_id in os.listdir(video_dir_path) 
                           if os.path.splitext(os.path.join(self.video_dir_path, video_id))[1] == '.mp4']

        if subset is not None:
            self.videos_ids = [video_id for video_id in self.videos_ids if video_id in subset]

    def __len__(self):
        return len(self.videos_ids)

    def __getitem__(self, idx):
        video_id = self.videos_ids[idx]
        video_path = os.path.join(self.video_dir_path, video_id)
        audio_path = os.path.join(self.audio_dir_path, video_id)

        waveform_and_sr = torchaudio.load(f"{audio_path}{self.audio_file_extension}")
        decord_vr = VideoReader(f"{video_path}{self.video_file_extension}", ctx=cpu(0))

        return decord_vr, waveform_and_sr, video_id

# if __name__ == '__main__':
#     with open("/home/lab/eitanshaar/zeroshot_AVE/test_AVE.txt", "r") as f:
#         test_files = f.read().strip().split("\n")
    
#     with open("/cortex/data/video/AVE/Annotations.txt", 'r') as f:
#         ave_data = f.read().strip().split("\n")[1:]
    
#     final_filenames = {}
#     for sample in ave_data:
#         sample = sample.split("&")
#         if sample[1] in test_files:
#             if sample[1] not in final_filenames:
#                 final_filenames[sample[1]] = []
#             final_filenames[sample[1]].append({'class': sample[0].replace("_", " ").lower(),
#                                                 'start': int(sample[3]),
#                                                 'end': int(sample[4])})
#     import json
#     with open("test_AVE.json", "w") as f:
#         json.dump(final_filenames, f)