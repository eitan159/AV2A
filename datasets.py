from torch.utils.data import Dataset
import os
from moviepy.editor import VideoFileClip
import torch
import random
from PIL import Image
import os
import numpy as np
from utils.video_pp import suppress_output
import torchaudio
import decord
from decord import VideoReader, cpu

torchaudio.set_audio_backend("soundfile")
decord.bridge.set_bridge('torch')


import random

def random_choose_indices(indices, step_size):
    """
    Randomly choose one index from each interval of the given step size.

    Args:
        indices (list or np.array): Array of indices (e.g., frame numbers).
        step_size (int): The size of each step/interval from which one index will be randomly chosen.

    Returns:
        list: List of randomly chosen indices.
    """
    chosen_indices = []
    # Loop through indices in steps of `step_size`
    for i in range(0, len(indices), step_size):
        # Define the interval
        interval = indices[i:i + step_size]
        # Randomly select one index from the interval
        if interval:  # Ensure interval is not empty
            chosen_index = random.choice(interval)
            chosen_indices.append(chosen_index)
    
    return chosen_indices

class AVE(Dataset):
    def __init__(self, video_dir_path, annotations_file_path, frames_transforms=None, 
                 sample_audio_sec=2) -> None:
        self.video_dir_path = video_dir_path
        self.frames_transforms = frames_transforms
        self.sample_audio_sec = sample_audio_sec
        self.audio_dir = "sample_audio_chunks"
        self.videos_ids = [video_id for video_id in os.listdir(video_dir_path) 
                           if os.path.splitext(os.path.join(self.video_dir_path, video_id))[1] == '.mp4']
        

        self.video_annotation_dict = {}
        with open(annotations_file_path, "r") as f:
            data = f.read().strip().split("\n")
        
        self.class2idx = {}
        for annotation in data:
            # Category&VideoID&Quality&StartTime&EndTime - example line in the annotations file
            category, video_id, _, start, end = annotation.split("&") 
            if category not in self.class2idx:
                self.class2idx[category] = len(self.class2idx)

            self.video_annotation_dict[f"{video_id}.mp4"] = [{'class_name': category,
                                                    'class_idx': self.class2idx[category],
                                                    'start': int(start),
                                                    'end': int(end)}]
        
        self.videos_ids = [video_id for video_id in self.videos_ids if video_id in self.video_annotation_dict]
        
    def __len__(self):
        return len(self.videos_ids)

    def __getitem__(self, idx):
        video_id = self.videos_ids[idx]
        video_path = os.path.join(self.video_dir_path, video_id)
        video = VideoFileClip(video_path) 

        audio = video.audio

        # Save the audio to a file
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
        remove_files_from_dir(self.audio_dir)

        for i, t in enumerate(range(0, int(audio.duration), self.sample_audio_sec)):
            audio_chunk = audio.subclip(t_start=t, t_end=t + self.sample_audio_sec)
            suppressed_write_audiofile = suppress_output(audio_chunk.write_audiofile)
            suppressed_write_audiofile(f"{self.audio_dir}/output_segment_{i}.wav")            

        frames = []
        fps = int(video.fps)
        for t in range(int(video.duration)):
            start_frame = random.randint(1, fps - 1)
            frame = video.get_frame(t + (start_frame / fps))
            frame = Image.fromarray(frame) # not good but for now it's ok
            if self.frames_transforms:
                frame = self.frames_transforms(frame)

            frames.append(frame)

        frames = torch.stack(frames)

        return frames, self.audio_dir, self.video_annotation_dict[video_id], video_id

class LLP(Dataset):
    def __init__(self, video_dir_path, audio_dir_path, frames_transforms=None, 
                 sample_audio_sec=2) -> None:
        self.video_dir_path = video_dir_path
        self.audio_dir_path = audio_dir_path
        self.frames_transforms = frames_transforms
        self.sample_audio_sec = sample_audio_sec
        self.audio_dir = "sample_audio_chunks"
        self.videos_ids = [video_id.replace(".mp4", "") for video_id in os.listdir(video_dir_path) 
                           if os.path.splitext(os.path.join(self.video_dir_path, video_id))[1] == '.mp4']

        
    def __len__(self):
        return len(self.videos_ids)

    def __getitem__(self, idx):
        video_id = self.videos_ids[idx]
        video_path = os.path.join(self.video_dir_path, video_id)
        audio_path = os.path.join(self.audio_dir_path, video_id)
        
        try:
            video = VideoFileClip(video_path)
        except:
            print(video_path)

        waveform_and_sr = torchaudio.load(audio_path)    

        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr) // 10 # 10 sec videos
        num_frames = 8 # language bind param 
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        

        return video_data, waveform_and_sr, video_id