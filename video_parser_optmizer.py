import torch
import numpy as np
from models.languagebindmodel.languagebind import to_device
from data_transforms import VisionTransform, AudioTransform
from label_shift import estimate_labelshift_ratio

class VideoParserOptimizer():
    def __init__(self, method, model, tokenizer, labels, device, sample_audio_sec, alpha,
                 filter_threshold, threshold_stage1, threshold_stage2, gamma, without_filter_classes,
                 without_refine_segments, dataset) -> None:
        self.method = method
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.device = device
        self.vision_transforms = VisionTransform()
        self.audio_transforms = AudioTransform()
        
        self.sample_audio_sec = sample_audio_sec
        self.alpha = alpha
        self.threshold_stage1 = threshold_stage1
        self.threshold_stage2 = threshold_stage2
        self.filter_threshold = filter_threshold
        self.gamma = gamma

        self.without_filter_classes = without_filter_classes
        self.without_refine_segments = without_refine_segments
        self.dataset = dataset
        
    def convert_results(self, results, video_id):
        coverted_results = []
        for event_label, event_segments in results.items():
            for segment in event_segments:
                coverted_results.append({
                    'event_label': event_label.lower(),
                    'start': segment[0],
                    'end': segment[1],
                })

        return {video_id: coverted_results}

    def optimize(self, similarities, decord_vr, waveform_and_sr, labels, video_id, similarity_type, embeddings):
        image_events_dict = {}
        thresholds = np.full((similarities.shape[0], len(labels)), self.threshold_stage1)
        count_events = np.zeros(len(labels))
        for event_dim in range(similarities.shape[0] - 1):
            tensor_slice_np = similarities[event_dim].cpu().numpy()
            indices = np.where(tensor_slice_np > thresholds[event_dim])[0]
            events = [labels[i] for i in indices]
            if self.method == "BBSE":
                count_events[indices] += 1
                label_shift_ratio = estimate_labelshift_ratio((similarities[:event_dim+1].cpu().numpy() > thresholds[:event_dim+1])*1, similarities[:event_dim+1].cpu().numpy(), 
                                                                np.expand_dims(similarities[event_dim + 1].cpu().numpy(), 0), len(labels))

                vector1 = embeddings['image'][event_dim]
                vector2 = embeddings['image'][event_dim + 1]

                cosine_similarity = torch.dot(vector1, vector2) / (torch.norm(vector1) * torch.norm(vector2))
                cosine_similarity = torch.clamp(cosine_similarity, min=0, max=1)

                offset = (self.threshold_stage1 * np.e**(-self.gamma*count_events)) * label_shift_ratio * cosine_similarity

                thresholds[event_dim + 1] = thresholds[event_dim] - offset
        
            
            image_events_dict[f"frame-{event_dim}"] = events


        tensor_slice_np = similarities[len(similarities) - 1].cpu().numpy()
        indices = np.where(tensor_slice_np > thresholds[len(similarities) - 1])[0]
        events = [labels[i] for i in indices]
        image_events_dict[f"frame-{len(similarities) - 1}"] = events


        results = self.optimize_video_events(image_events_dict)
        results = self.increment_end_times(results)
        results = self.merge_consecutive_segments(results)
        results = self.filter_events(results)
        
        if not self.without_refine_segments:
            results = self.refine_segments(results, decord_vr, waveform_and_sr, labels, similarity_type)
        
        return self.convert_results(results, video_id)

    def predict(self, labels, decord_vr, waveform_and_sr, video_id):
        if not self.without_filter_classes:
            combined_filtered_labels, video_filtered_labels, audio_filtered_labels = self.filter_classes(labels, decord_vr, waveform_and_sr, self.filter_threshold)
        else:
            combined_filtered_labels, video_filtered_labels, audio_filtered_labels = labels, labels, labels
        
        inputs = {
            'image': {"pixel_values": self.vision_transforms(decord_vr, transform_type='image').to(self.device)},
            'audio': {"pixel_values": self.audio_transforms.split_sample_audio(waveform_and_sr, self.sample_audio_sec).to(self.device)},
        }
        combined_similarities, _, _, embeddings_combined = self.get_similiraties(combined_filtered_labels, inputs, self.alpha)
        _, video_text_similarties, _, embeddings_video_text = self.get_similiraties(video_filtered_labels, inputs, self.alpha)
        _, _, audio_text_similarites, embeddings_audio_text = self.get_similiraties(audio_filtered_labels, inputs, self.alpha)

        combined_results = self.optimize(combined_similarities, decord_vr, waveform_and_sr, combined_filtered_labels, video_id, "combined", embeddings_combined)
        video_results = self.optimize(video_text_similarties, decord_vr, waveform_and_sr, video_filtered_labels, video_id, "video", embeddings_video_text)
        audio_results = self.optimize(audio_text_similarites, decord_vr, waveform_and_sr, audio_filtered_labels, video_id, "audio", embeddings_audio_text)

        return combined_results, video_results, audio_results

    def optimize_video_events(self, image_events_dict):
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

    def increment_end_times(self, events):
        updated_events = {}
        for key, time_intervals in events.items():
            updated_intervals = []
            for interval in time_intervals:
                if len(interval) == 2:
                    start, end = interval
                    updated_intervals.append([start, end + 1])
            updated_events[key] = updated_intervals
        return updated_events

    def filter_events(self, events_dict):
        filtered_events = {}
        for key, intervals in events_dict.items():
            filtered_intervals = [interval for interval in intervals if len(interval) == 2 and interval[1] - interval[0] > 1]
            if filtered_intervals:
                filtered_events[key] = filtered_intervals
        
        return filtered_events

    def merge_consecutive_segments(self, events):
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

    def refine_segments(self, video_events, decord_vr, waveform_and_sr, labels, similarity_type):
        if (len(video_events) == 1 and len(next(iter(video_events.values()))) == 1) or (not video_events):
            return video_events
        else:
            refined_segments = {}
            past_segments = {}
            for event, time_ranges in video_events.items():
                for start_time, end_time in time_ranges:

                    inputs = {
                        'video': {"pixel_values": self.vision_transforms(decord_vr, transform_type='video', start=start_time, end=end_time).unsqueeze(0).to(self.device)},
                        'audio': {"pixel_values": self.audio_transforms(waveform_and_sr, start_sec=start_time, end_sec=end_time).unsqueeze(0).to(self.device)},
                    }
                    combined_similarities, video_text_similarties, audio_text_similarties, _ = self.get_similiraties(labels, inputs, self.alpha)
                    
                    if similarity_type == "combined":
                        if self.dataset == "AVE":
                            events = [labels[combined_similarities.argmax().item()]]
                        else:    
                            events = self.events_above_threshold(labels, combined_similarities, self.threshold_stage2)
                        similarities = combined_similarities[0]
                    elif similarity_type == "video":
                        if self.dataset == "AVE":
                            events = [labels[video_text_similarties.argmax().item()]]
                        else:  
                            events = self.events_above_threshold(labels, video_text_similarties, self.threshold_stage2)
                        similarities = video_text_similarties[0]
                    else:
                        if self.dataset == "AVE":
                            events = [labels[audio_text_similarties.argmax().item()]]
                        else:  
                            events = self.events_above_threshold(labels, audio_text_similarties, self.threshold_stage2)
                        similarities = audio_text_similarties[0]

                    if event in events:
                        if event in refined_segments:
                            refined_segments[event].append((start_time, end_time))
                        else:
                            refined_segments[event] = [(start_time, end_time)]
                    
                    if event in past_segments:
                        past_segments[event].append(((start_time, end_time), similarities[labels.index(event)]))
                    else:
                        past_segments[event] = [((start_time, end_time), similarities[labels.index(event)])]
            
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
                    
    def get_similiraties(self, labels, inputs, alpha):
        preprocessed_labels = [f"A {label.replace('_', ' ').lower()}" for label in labels]

        inputs['language'] = to_device(self.tokenizer(preprocessed_labels, max_length=77, padding='max_length',
                                                                truncation=True, return_tensors='pt'), self.device)

        with torch.no_grad():
            embeddings = self.model(inputs)
        
        if 'image' in embeddings:
            if embeddings['image'].shape[0] != embeddings['audio'].shape[0]:
                embeddings['audio'] = embeddings['audio'].repeat_interleave(self.sample_audio_sec, dim=0)
            
            video_text_similarity = embeddings['image'] @ embeddings['language'].T   
        
        else:
            video_text_similarity = embeddings['video'] @ embeddings['language'].T   
        
        audio_text_similarity = embeddings['audio'] @ embeddings['language'].T
            
        if video_text_similarity.shape[0] < audio_text_similarity.shape[0]:
            pad = torch.zeros_like(video_text_similarity[0].unsqueeze(0)).to(self.device)
            pad = pad.repeat_interleave(audio_text_similarity.shape[0] - video_text_similarity.shape[0])
            video_text_similarity = torch.cat((video_text_similarity, pad), dim=0)
        elif video_text_similarity.shape[0] > audio_text_similarity.shape[0]:
            pad = torch.zeros_like(audio_text_similarity[0].unsqueeze(0)).to(self.device)
            pad = pad.repeat(video_text_similarity.shape[0] - audio_text_similarity.shape[0], 1)
            audio_text_similarity = torch.cat((audio_text_similarity, pad), dim=0)
        
        video_text_similarity_norm = (video_text_similarity - torch.mean(video_text_similarity, dim=-1, keepdim=True)) / torch.std(video_text_similarity, dim=-1, keepdim=True)
        video_text_similarity_sigmoid_norm = torch.sigmoid(video_text_similarity_norm)

        audio_text_similarity_norm = (audio_text_similarity - torch.mean(audio_text_similarity, dim=-1, keepdim=True)) / torch.std(audio_text_similarity, dim=-1, keepdim=True)
        audio_text_similarity_sigmoid_norm = torch.sigmoid(audio_text_similarity_norm)

        if alpha > 0:
            combined_similarities = (1 - alpha) * video_text_similarity_sigmoid_norm + (alpha) * audio_text_similarity_sigmoid_norm
        else:
            combined_similarities = video_text_similarity_sigmoid_norm * audio_text_similarity_sigmoid_norm

        return combined_similarities, video_text_similarity_sigmoid_norm, audio_text_similarity_sigmoid_norm, embeddings


    def filter_classes(self, labels, decord_vr, waveform_and_sr, filter_threshold):
        inputs = {
            'video': {"pixel_values": self.vision_transforms(decord_vr, transform_type='video').unsqueeze(0).to(self.device)},
            'audio': {"pixel_values": self.audio_transforms(waveform_and_sr).unsqueeze(0).to(self.device)},
        }
        
        combined_similarities, video_text_similarity, audio_text_similarity, _ = self.get_similiraties(labels, inputs, self.alpha)

        combined_events = self.events_above_threshold(labels, combined_similarities, filter_threshold)
        video_events = self.events_above_threshold(labels, video_text_similarity, filter_threshold)
        audio_events = self.events_above_threshold(labels, audio_text_similarity, filter_threshold)

        return combined_events, video_events, audio_events

    def events_above_threshold(self, labels, similarities, threshold):
        indices = torch.nonzero(similarities > threshold, as_tuple=False)
        col_indices = indices[:, 1].tolist()
        events = [labels[i] for i in col_indices]

        return events