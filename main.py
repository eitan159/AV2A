import json
from dataset import LLP
import torch
from tqdm import tqdm
import argparse
import numpy as np
import os
from models.languagebindmodel.languagebind import LanguageBind, LanguageBindImageTokenizer, to_device
from data_transforms import VisionTransform, AudioTransform


def convert_results(results, video_id):
    coverted_results = []
    for event_label, event_segments in results.items():
        for segment in event_segments:
            coverted_results.append({
                'event_label': event_label.lower(),
                'start': segment[0],
                'end': segment[1],
            })

    return {video_id: coverted_results}

def optimize(similarities, decord_vr, waveform_and_sr, labels, video_id, similarity_type):
    image_events_dict = {}
    for event_dim in range(similarities.shape[0]):
        tensor_slice_np = similarities[event_dim].cpu().numpy()
        indices = np.where(tensor_slice_np > threshold_stage1)[0]
        events = [labels[i] for i in indices]
        image_events_dict[f"frame-{event_dim}"] = events


    results = optimize_video_events(image_events_dict)
    results = increment_end_times(results)
    results = merge_consecutive_segments(results)
    results = filter_events(results)
    results = refine_segments(results, decord_vr, waveform_and_sr, labels, similarity_type)
    
    return convert_results(results, video_id)

def predict(labels, decord_vr, waveform_and_sr, video_id):
    combined_filtered_labels, video_filtered_labels, audio_filtered_labels = filter_classes(labels, decord_vr, waveform_and_sr, filter_threshold)
    
    inputs = {
        'image': {"pixel_values": vision_transforms(decord_vr, transform_type='image').to(device)},
        'audio': {"pixel_values": audio_transforms.split_sample_audio(waveform_and_sr, args.sample_audio_sec).to(device)},
    }
    combined_similarities, _, _ = get_similiraties(combined_filtered_labels, inputs, alpha)
    _, video_text_similarties, _ = get_similiraties(video_filtered_labels, inputs, alpha)
    _, _, audio_text_similarites = get_similiraties(audio_filtered_labels, inputs, alpha)

    combined_results = optimize(combined_similarities, decord_vr, waveform_and_sr, combined_filtered_labels, video_id, "combined")
    video_results = optimize(video_text_similarties, decord_vr, waveform_and_sr, video_filtered_labels, video_id, "video")
    audio_results = optimize(audio_text_similarites, decord_vr, waveform_and_sr, audio_filtered_labels, video_id, "audio")

    return combined_results, video_results, audio_results

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

def refine_segments(video_events, decord_vr, waveform_and_sr, labels, similarity_type):
    if (len(video_events) == 1 and len(next(iter(video_events.values()))) == 1) or (not video_events):
        return video_events
    else:
        refined_segments = {}
        past_segments = {}
        for event, time_ranges in video_events.items():
            for start_time, end_time in time_ranges:

                inputs = {
                    'video': {"pixel_values": vision_transforms(decord_vr, transform_type='video', start=start_time, end=end_time).unsqueeze(0).to(device)},
                    'audio': {"pixel_values": audio_transforms(waveform_and_sr, start_sec=start_time, end_sec=end_time).unsqueeze(0).to(device)},
                }
                combined_similarities, video_text_similarties, audio_text_similarties = get_similiraties(labels, inputs, alpha)
                
                # TODO use threshold_stage 2 here !!!
                if similarity_type == "combined":
                    max_value, max_index = torch.max(combined_similarities, dim=1)
                elif similarity_type == "video":
                    max_value, max_index = torch.max(video_text_similarties, dim=1)
                else:
                    max_value, max_index = torch.max(audio_text_similarties, dim=1)


                if labels[max_index] == event:
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
                 
def get_similiraties(labels, inputs, alpha):
    preprocessed_labels = [f"A {label.replace('_', ' ').lower()}" for label in labels]

    inputs['language'] = to_device(tokenizer(preprocessed_labels, max_length=77, padding='max_length',
                                                            truncation=True, return_tensors='pt'), device)

    with torch.no_grad():
        embeddings = model(inputs)
    
    if 'image' in embeddings:        
        if embeddings['image'].shape[0] != embeddings['audio'].shape[0]:
            embeddings['audio'] = embeddings['audio'].repeat_interleave(args.sample_audio_sec, dim=0)
        
        video_text_similarity = embeddings['image'] @ embeddings['language'].T   
    
    else:
        video_text_similarity = embeddings['video'] @ embeddings['language'].T   
    
    audio_text_similarity = embeddings['audio'] @ embeddings['language'].T
        
    if video_text_similarity.shape[0] < audio_text_similarity.shape[0]:
        pad = torch.zeros_like(video_text_similarity[0].unsqueeze(0)).to(device)
        pad = pad.repeat_interleave(audio_text_similarity.shape[0] - video_text_similarity.shape[0])
        video_text_similarity = torch.cat((video_text_similarity, pad), dim=0)
    elif video_text_similarity.shape[0] > audio_text_similarity.shape[0]:
        pad = torch.zeros_like(audio_text_similarity[0].unsqueeze(0)).to(device)
        pad = pad.repeat(video_text_similarity.shape[0] - audio_text_similarity.shape[0], 1)
        audio_text_similarity = torch.cat((audio_text_similarity, pad), dim=0)
    
    video_text_similarity_norm = (video_text_similarity - torch.mean(video_text_similarity)) / torch.std(video_text_similarity)
    video_text_similarity_sigmoid_norm = torch.sigmoid(video_text_similarity_norm)

    audio_text_similarity_norm = (audio_text_similarity - torch.mean(audio_text_similarity)) / torch.std(audio_text_similarity)
    audio_text_similarity_sigmoid_norm = torch.sigmoid(audio_text_similarity_norm)

    combined_similarities = (1 - alpha) * video_text_similarity_sigmoid_norm + (alpha) * audio_text_similarity_sigmoid_norm

    return combined_similarities, video_text_similarity_sigmoid_norm, audio_text_similarity_sigmoid_norm


def filter_classes(labels, decord_vr, waveform_and_sr, filter_threshold):
    inputs = {
        'video': {"pixel_values": vision_transforms(decord_vr, transform_type='video').unsqueeze(0).to(device)},
        'audio': {"pixel_values": audio_transforms(waveform_and_sr).unsqueeze(0).to(device)},
    }
    
    combined_similarities, video_text_similarity, audio_text_similarity = get_similiraties(labels, inputs, alpha)
    
    indices = torch.nonzero(combined_similarities > filter_threshold, as_tuple=False)
    col_indices = indices[:, 1].tolist()
    combined_events = [labels[i] for i in col_indices]

    indices = torch.nonzero(video_text_similarity > filter_threshold, as_tuple=False)
    col_indices = indices[:, 1].tolist()
    video_events = [labels[i] for i in col_indices]

    indices = torch.nonzero(audio_text_similarity > filter_threshold, as_tuple=False)
    col_indices = indices[:, 1].tolist()
    audio_events = [labels[i] for i in col_indices]

    return combined_events, video_events, audio_events

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', required=True, type=str)    
    parser.add_argument('--audio_dir_path', required=True, type=str)
    parser.add_argument('--gpu_id', default=-1, type=int)
    parser.add_argument('--threshold_stage1', default=0.6, type=float)
    parser.add_argument('--threshold_stage2', default=0.6, type=float)
    parser.add_argument('--filter_threshold', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--candidates_file_path', required=True, type=str)
    parser.add_argument('--sample_audio_sec', default=2, type=int)
    parser.add_argument('--output_video_path', default="./cropped_files/cropped_video.mp4", type=str)
    parser.add_argument('--output_audio_path', default="./cropped_files/cropped_audio.wav", type=str)
    args = parser.parse_args()

    threshold_stage1 = args.threshold_stage1
    threshold_stage2 = args.threshold_stage2

    filter_threshold = args.filter_threshold
    alpha = args.alpha
    video_dir_path = args.video_dir_path
    output_video_path = args.output_video_path
    output_audio_path = args.output_audio_path
    
    dataset = LLP(args.video_dir_path,
                  args.audio_dir_path)
    
    vision_transforms = VisionTransform()
    audio_transforms = AudioTransform()
    
    labels = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']

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
    # modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    combined_candidates, video_candidates, audio_candidates = [], [], []
    for sample in tqdm(dataset, desc="Processing samples"):
        decord_vr, waveform_and_sr, video_id = sample
        combined_results, video_results, audio_results = predict(labels, decord_vr, waveform_and_sr, video_id)
        
        combined_candidates.append(combined_results)
        video_candidates.append(video_results)
        audio_candidates.append(audio_results)
    
    with open(args.candidates_file_path, 'w') as f:
        json.dump({
            "combined": combined_candidates,
            "video": video_candidates,
            "audio": video_candidates}, f)
