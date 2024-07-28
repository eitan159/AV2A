from torch.utils.data import Dataset
import os
from moviepy.editor import VideoFileClip
import torch
import random
from PIL import Image
import os

from utils.video_pp import suppress_output

def remove_files_from_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

class AVE(Dataset):
    def __init__(self, video_dir_path, annotations_file_path, frames_transforms=None) -> None:
        self.video_dir_path = video_dir_path
        self.frames_transforms = frames_transforms
        self.audio_dir = "sample_audio_chunks"
        self.videos_ids = [video_id for video_id in os.listdir(video_dir_path) 
                           if os.path.splitext(os.path.join(self.video_dir_path, video_id))[1] == '.mp4']
        

        self.video_annotation_dict = {}
        with open(annotations_file_path, "r") as f:
            data = f.read().strip().split("\n")
        
        self.class2idx = {}
        for annotation in data[1:]:
            # Category&VideoID&Quality&StartTime&EndTime - example line in the annotations file
            category, video_id, _, start, end = annotation.split("&") 
            if category not in self.class2idx:
                self.class2idx[category] = len(self.class2idx)

            self.video_annotation_dict[f"{video_id}.mp4"] = {'class_name': category,
                                                    'class_idx': self.class2idx[category],
                                                    'start': int(start),
                                                    'end': int(end)}
        
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

        for i, t in enumerate(range(0, int(audio.duration), 2)):
            audio_chunk = audio.subclip(t_start=t, t_end=t + 2)
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
