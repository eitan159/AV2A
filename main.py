import json
from calc_metric import calc_metrics
import torch
from tqdm import tqdm
from dataset import AVE
import argparse
import numpy as np
from data_transforms import language_bind_transform
import os
from models.languagebindmodel.languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from utils.video_pp import crop_video_and_extract_audio, extract_audio, get_video_duration


def predict(labels, frames, audio_files, video_id):

    preprocessed_labels = [f"A {label.split(',')[0]}" for label in labels]

    inputs = {
        'image': {"pixel_values": frames.to(device)},
        'audio': to_device(modality_transform['audio'](audio_files), device),
    }
    inputs['language'] = to_device(tokenizer(preprocessed_labels, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)

    with torch.no_grad():
        embeddings = model(inputs)
    
    embeddings['audio'] = embeddings['audio'].repeat_interleave(2, dim=0)

    video_text_similarity = embeddings['image'] @ embeddings['language'].T   
    audio_text_similarity = embeddings['audio'] @ embeddings['language'].T
    # vision_audio_similarity = embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T

    video_text_similarity = torch.softmax(video_text_similarity, dim=-1)
    audio_text_similarity = torch.softmax(audio_text_similarity, dim=-1)
    # alphas = torch.softmax(vision_audio_similarity, dim=-1).diagonal().unsqueeze(1)

    similarities = alpha * video_text_similarity + (1 - alpha) * audio_text_similarity
    similarities = torch.softmax(similarities, dim=-1)

    # similarities = torch.softmax(similarities, dim=-1)
    
    image_events_dict = {}
    for event_dim in range(similarities.shape[0]):
        tensor_slice_np = similarities[event_dim].cpu().numpy()
        indices = np.where(tensor_slice_np > threshold)[0]
        events = [labels[i] for i in indices]
        # events = similarities[event_dim].argmax()
        # values = tensor_slice_np[indices]
        image_events_dict[f"frame-{event_dim}"] = events


    results = optimize_video_events(image_events_dict)
    results = increment_end_times(results)
    results = merge_consecutive_segments(results)
    results = filter_events(results)
    results = refine_segments(results, video_id, preprocessed_labels)
    # refined_video_events = stage3(refined_video_events, video_id, preprocessed_labels)
    
    return results


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

def increment_end_times(events):
    updated_events = {}
    for key, time_intervals in events.items():
        updated_intervals = []
        for interval in time_intervals:
            if len(interval) == 2:
                start, end = interval
                updated_intervals.append([start, end + 1])
        updated_events[key] = updated_intervals
    return updated_events

def filter_events(events_dict):
    filtered_events = {}
    for key, intervals in events_dict.items():
        filtered_intervals = [interval for interval in intervals if len(interval) == 2 and interval[1] - interval[0] > 1]
        if filtered_intervals:
            filtered_events[key] = filtered_intervals
    
    return filtered_events


def merge_consecutive_segments(events):
    updated_events = {}
    for key, time_intervals in events.items():
        if not time_intervals:
            updated_events[key] = []
            continue

        # Sort the intervals to ensure they are in order
        sorted_intervals = sorted(time_intervals)
        merged_intervals = [sorted_intervals[0]]

        for current in sorted_intervals[1:]:
            last = merged_intervals[-1]
            if current[0] <= last[1] + 1:  # Check if intervals are consecutive or overlapping
                last[1] = max(last[1], current[1])  # Merge intervals
            else:
                merged_intervals.append(current)  # Add new interval

        updated_events[key] = merged_intervals

    return updated_events


def refine_segments(video_events, video_id, preprocessed_labels):
    if (len(video_events) == 1 and len(next(iter(video_events.values()))) == 1) or (not video_events):
        return video_events
    else:
        refined_segments = {}
        past_segments = {}
        video_path = os.path.join(video_dir_path, video_id)
        for event, time_ranges in video_events.items():
            for start_time, end_time in time_ranges:
                crop_video_and_extract_audio(video_path, start_time, end_time, output_video_path, output_audio_path)

                inputs = {
                    'video': to_device(modality_transform['video']([output_video_path]), device),
                    'audio': to_device(modality_transform['audio']([output_audio_path]), device),
                }
                inputs['language'] = to_device(tokenizer(preprocessed_labels, max_length=77, padding='max_length',
                                                        truncation=True, return_tensors='pt'), device)

                with torch.no_grad():
                    embeddings = model(inputs)
                
                # embeddings['audio'] = embeddings['audio'].repeat_interleave(2, dim=0)

                video_text_similarity = embeddings['video'] @ embeddings['language'].T   
                audio_text_similarity = embeddings['audio'] @ embeddings['language'].T

                video_text_similarity = torch.softmax(video_text_similarity, dim=-1)
                audio_text_similarity = torch.softmax(audio_text_similarity, dim=-1)

                similarities = (1 - alpha) * video_text_similarity + (alpha) * audio_text_similarity
                similarities = torch.softmax(similarities, dim=-1)
                max_value, max_index = torch.max(similarities, dim=1)
                
                if preprocessed_labels[max_index] == f"A {event.split(',')[0]}":
                    if event in refined_segments:
                        refined_segments[event].append((start_time, end_time))
                    else:
                        refined_segments[event] = [(start_time, end_time)]
                
                if event in past_segments:
                    past_segments[event].append(((start_time, end_time), max_value))
                else:
                    past_segments[event] = [((start_time, end_time), max_value)]
        
        if not refined_segments:
            max_score = float('-inf')
            max_event = None
            max_times = None
            for event, segments in past_segments.items():
                for segment in segments:
                    if isinstance(segment, tuple) and len(segment) == 2:
                        times, score = segment
                        if score.item() > max_score:
                            max_score = score.item()
                            max_event = event
                            max_times = times

            return {max_event: [list(max_times)]}

        return refined_segments
    
def stage3(video_events, video_id, preprocessed_labels):
    refined_segments = {}
    video_path = os.path.join(video_dir_path, video_id)
    video_length = round(get_video_duration(video_path))
    for event, time_ranges in video_events.items():
        for start_time, end_time in time_ranges:
            refined_left_value = start_time
            refined_right_value = end_time
            extract_audio(video_path, output_audio_path)
            og_similarities = get_similiraties(preprocessed_labels, video_path, output_audio_path)
            og_max_value, _ = torch.max(og_similarities, dim=1)    
                            
            if start_time - 1 >= 0:
                crop_video_and_extract_audio(video_path, start_time - 1, end_time, output_video_path, output_audio_path)
                similarities = get_similiraties(preprocessed_labels, output_video_path, output_audio_path)
                max_value, max_index = torch.max(similarities, dim=1)
                    
                if (max_value > og_max_value + 0.001) and (preprocessed_labels[max_index] == f"A {event.split(',')[0]}"):
                    refined_left_value = start_time - 1
                    og_max_value = max_value
                
            if start_time + 1 <= video_length:     
                crop_video_and_extract_audio(video_path, start_time + 1, end_time, output_video_path, output_audio_path)
                similarities = get_similiraties(preprocessed_labels, output_video_path, output_audio_path)
                max_value, max_index = torch.max(similarities, dim=1)
                    
                if (max_value > og_max_value + 0.001) and (preprocessed_labels[max_index] == f"A {event.split(',')[0]}"):
                    refined_left_value = start_time + 1                    



            if end_time - 1 >= 0:
                crop_video_and_extract_audio(video_path, start_time, end_time - 1, output_video_path, output_audio_path)
                similarities = get_similiraties(preprocessed_labels, output_video_path, output_audio_path)
                max_value, max_index = torch.max(similarities, dim=1)
                    
                if (max_value > og_max_value + 0.001) and (preprocessed_labels[max_index] == f"A {event.split(',')[0]}"):
                    refined_right_value = end_time - 1
                    og_max_value = max_value
                
            if end_time + 1 <= video_length:     
                crop_video_and_extract_audio(video_path, start_time, end_time + 1, output_video_path, output_audio_path)
                similarities = get_similiraties(preprocessed_labels, output_video_path, output_audio_path)
                max_value, max_index = torch.max(similarities, dim=1)
                    
                if (max_value > og_max_value + 0.001) and (preprocessed_labels[max_index] == f"A {event.split(',')[0]}"):
                    refined_right_value = end_time + 1   

            if event in refined_segments:
                refined_segments[event].append((refined_left_value, refined_right_value))
            else:
                refined_segments[event] = [(refined_left_value, refined_right_value)]
                                        
            # print(refined_left_value, refined_right_value)   
            
    return refined_segments                       



                    
def get_similiraties(preprocessed_labels, output_video_path, output_audio_path):
    inputs = {
        'video': to_device(modality_transform['video']([output_video_path]), device),
        'audio': to_device(modality_transform['audio']([output_audio_path]), device),
    }
    inputs['language'] = to_device(tokenizer(preprocessed_labels, max_length=77, padding='max_length',
                                                            truncation=True, return_tensors='pt'), device)

    with torch.no_grad():
        embeddings = model(inputs)
                    

    video_text_similarity = embeddings['video'] @ embeddings['language'].T   
    audio_text_similarity = embeddings['audio'] @ embeddings['language'].T

    video_text_similarity = torch.softmax(video_text_similarity, dim=-1)
    audio_text_similarity = torch.softmax(audio_text_similarity, dim=-1)

    similarities = (1 - alpha) * video_text_similarity + (alpha) * audio_text_similarity
    similarities = torch.softmax(similarities, dim=-1)
    return similarities
                    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', required=True, type=str)
    parser.add_argument('--annotations_file_path', required=True, type=str)
    parser.add_argument('--gpu_id', default=-1, type=int)
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--candidates_file_path', required=True, type=str)
    parser.add_argument('--sample_audio_sec', default=2, type=int)
    parser.add_argument('--output_video_path', default="./cropped_files/cropped_video.mp4", type=str)
    parser.add_argument('--output_audio_path', default="./cropped_files/cropped_audio.wav", type=str)
    args = parser.parse_args()

    threshold = args.threshold
    alpha = args.alpha
    video_dir_path = args.video_dir_path
    output_video_path = args.output_video_path
    output_audio_path = args.output_audio_path
    
    dataset = AVE(args.video_dir_path,
                  args.annotations_file_path,
                  frames_transforms=language_bind_transform,
                  sample_audio_sec=args.sample_audio_sec)
    
    labels = list(dataset.class2idx.keys())

    device = f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cpu"


    clip_type = {
        'video': 'LanguageBind_Video_FT', 
        'audio': 'LanguageBind_Audio_FT',
        'image': 'LanguageBind_Image',
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'lb203/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}


    candidates = {}
    for sample in tqdm(dataset, desc="Processing samples"):
        frames, audio_dir, label_dict, video_id = sample
        video_id_withot_extention = video_id.replace('.mp4', '')
        audio_paths = [f"{dataset.audio_dir}/{file_name}" for file_name in os.listdir(dataset.audio_dir)]
        video_events = predict(labels, frames, audio_paths, video_id)
        candidates[video_id_withot_extention] = video_events
    
    with open(args.candidates_file_path, 'w') as f:
        json.dump(candidates, f)
    calc_metrics(args.annotations_file_path, args.candidates_file_path)
