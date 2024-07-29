import json
from calc_metric import calc_metrics
from models.imagebindmodel.imagebind.data import load_and_transform_text, load_and_transform_audio_data
import torch
from tqdm import tqdm
from models.imagebindmodel.imagebind.models import imagebind_model
from models.imagebindmodel.imagebind.models.imagebind_model import ModalityType
from dataset import AVE
import argparse
import numpy as np
from data_transforms import image_transforms_imagebind
import os

def predict(labels, frames, audio_files):
    modality_inputs = {
        ModalityType.TEXT: load_and_transform_text(labels, device),
        ModalityType.VISION: frames.to(device),
        ModalityType.AUDIO: load_and_transform_audio_data(audio_files, device),
    }

    with torch.no_grad():
        embeddings = model(modality_inputs)
    
    video_text_similarity = embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T   
    audio_text_similarity = (embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T).repeat_interleave(2, dim=0)
    
    video_text_similarity = torch.softmax(video_text_similarity, dim=-1)
    audio_text_similarity = torch.softmax(audio_text_similarity, dim=-1)

    similarities = alpha * video_text_similarity + (1 - alpha) * audio_text_similarity
    similarities = torch.softmax(similarities, dim=-1)

    # similarities = torch.softmax(similarities, dim=-1)
    
    image_events_dict = {}
    for event_dim in range(similarities.shape[0]):
        tensor_slice_np = similarities[event_dim].cpu().numpy()
        indices = np.where(tensor_slice_np > threshold)[0]
        events = [labels[i] for i in indices]
        # values = tensor_slice_np[indices]
        image_events_dict[f"frame-{event_dim}"] = events


    unfiltered_video_events = optimize_video_events(image_events_dict)
    video_events = filter_events(unfiltered_video_events)
    return video_events


def optimize_video_events(image_events_dict):
    transformed_dict = {}
    sorted_frames = sorted(image_events_dict.keys(), key=lambda x: int(x.split('-')[1]))

    for frame in sorted_frames:
        frame_number = int(frame.split('-')[1])
        events = image_events_dict[frame]

        for event in events:
            if event not in transformed_dict:
                transformed_dict[event] = []
                
            if not transformed_dict[event] or transformed_dict[event][-1][1] != frame_number - 1:
                # start a new interval
                transformed_dict[event].append([frame_number, frame_number])
            else:
                # update the end of the current interval
                transformed_dict[event][-1][1] = frame_number

    # convert single frame intervals to a single number ####only for clarity!!!
    for event in transformed_dict:
        transformed_dict[event] = [
            [start, end] if start != end else [start] for start, end in transformed_dict[event]
        ]

    return transformed_dict


def filter_events(events_dict):
    filtered_events = {}
    for key, intervals in events_dict.items():
        filtered_intervals = [interval for interval in intervals if len(interval) == 2 and interval[1] - interval[0] > 1]
        if filtered_intervals:
            filtered_events[key] = filtered_intervals
    
    return filtered_events

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', required=True, type=str)
    parser.add_argument('--annotations_file_path', required=True, type=str)
    parser.add_argument('--gpu_id', default=-1, type=int)
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--candidates_file_path', required=True, type=str)
    args = parser.parse_args()

    threshold = args.threshold
    alpha = args.alpha
    
    dataset = AVE(args.video_dir_path,
                  args.annotations_file_path,
                  frames_transforms=image_transforms_imagebind)
    
    labels = list(dataset.class2idx.keys())

    device = f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    candidates = {}
    for sample in tqdm(dataset, desc="Processing samples"):
        frames, audio_dir, label_dict, video_id = sample
        video_id = video_id.replace('.mp4', '')
        audio_paths = [f"{dataset.audio_dir}/{file_name}" for file_name in os.listdir(dataset.audio_dir)]
        video_events = predict(labels, frames, audio_paths)
        candidates[video_id] = video_events
    
    with open(args.candidates_file_path, 'w') as f:
        json.dump(candidates, f)
    calc_metrics(args.annotations_file_path)

