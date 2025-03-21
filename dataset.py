from torch.utils.data import Dataset
import os
import numpy as np
import torchaudio
import decord
from decord import VideoReader, cpu
import librosa

torchaudio.set_audio_backend("soundfile")
decord.bridge.set_bridge('torch')


class VideoDataset(Dataset):
    def __init__(self, video_dir_path, audio_dir_path,
                 backbone,
                 video_file_extension = ".mp4", 
                 audio_file_extension = ".wav",
                 subset = None) -> None:
        
        self.video_dir_path = video_dir_path
        self.audio_dir_path = audio_dir_path
        self.video_file_extension = video_file_extension
        self.audio_file_extension = audio_file_extension
        self.videos_ids = [video_id.replace(".mp4", "") for video_id in os.listdir(video_dir_path) 
                           if os.path.splitext(os.path.join(self.video_dir_path, video_id))[1] == '.mp4']
        self.backbone = backbone

        if subset is not None:
            self.videos_ids = [video_id for video_id in self.videos_ids if video_id in subset]

    def __len__(self):
        return len(self.videos_ids)

    def __getitem__(self, idx):
        video_id = self.videos_ids[idx]
        video_path = os.path.join(self.video_dir_path, video_id)
        audio_path = os.path.join(self.audio_dir_path, video_id)
        
        if self.backbone == 'language_bind':
            waveform_and_sr = torchaudio.load(f"{audio_path}{self.audio_file_extension}")
        else:
            audio_data, sr = librosa.load(f"{audio_path}{self.audio_file_extension}", sr=48000)
            audio_data = np.expand_dims(audio_data, axis=0)
            waveform_and_sr = (audio_data, sr)

        decord_vr = VideoReader(f"{video_path}{self.video_file_extension}", ctx=cpu(0))

        return decord_vr, waveform_and_sr, video_id