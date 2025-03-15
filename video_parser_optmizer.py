import torch
import numpy as np
from data_transforms import VisionTransform, AudioTransform
from label_shift import estimate_labelshift_ratio
from backbones import LanguageBind_model, CLIP_CLAP_model
from torch.nn.functional import cosine_similarity

class VideoParserOptimizer():
    def __init__(self, method, backbone, labels, device, alpha,
                 filter_threshold, threshold_stage1, threshold_stage2, gamma, without_filter_classes,
                 without_refine_segments, dataset, fusion):
        
        self.method = method
        self.labels = labels
        self.device = device
        
        self.alpha = alpha
        self.threshold_stage1 = threshold_stage1
        self.threshold_stage2 = threshold_stage2
        self.filter_threshold = filter_threshold
        self.gamma = gamma

        self.without_filter_classes = without_filter_classes
        self.without_refine_segments = without_refine_segments
        self.dataset = dataset

        self.model = LanguageBind_model(device, alpha) if backbone == 'language_bind' else CLIP_CLAP_model(device, alpha)
        self.vision_transforms = VisionTransform(model=backbone)
        self.audio_transforms = AudioTransform(model=backbone)

        self.fusion = fusion


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

    def events_above_threshold(self, labels, similarities, threshold):
        indices = torch.nonzero(similarities > threshold, as_tuple=False)
        col_indices = indices[:, 1].tolist()
        events = [labels[i] for i in col_indices]

        return events

    def late_fusion(self, video_id, video_results, audio_results):
            # Convert results to structured NumPy arrays for fast processing
            video_events = np.array([(v['event_label'], v['start'], v['end']) for v in video_results[video_id]])
            audio_events = np.array([(a['event_label'], a['start'], a['end']) for a in audio_results[video_id]])

            if video_events.size == 0 or audio_events.size == 0:
                return {video_id: []}

            # Extract unique event labels that appear in both video and audio
            common_events = np.intersect1d(video_events[:, 0], audio_events[:, 0])

            late_fusion_av = []
            
            for event_label in common_events:
                # Select matching event segments for the given label
                video_segments = video_events[video_events[:, 0] == event_label][:, 1:].astype(int)
                audio_segments = audio_events[audio_events[:, 0] == event_label][:, 1:].astype(int)

                # Create a binary timeline for video & audio presence
                max_time = 10  # Assuming max timeline length (adjust if needed)
                pred_video = np.zeros(max_time, dtype=int)
                pred_audio = np.zeros(max_time, dtype=int)

                for start, end in video_segments:
                    pred_video[start:end] = 1
                for start, end in audio_segments:
                    pred_audio[start:end] = 1

                # Compute late fusion by logical AND
                pred_av = np.logical_and(pred_audio, pred_video).astype(int)

                # Detect event segments (vectorized)
                diff = np.diff(np.concatenate(([0], pred_av, [0])))  # Add 0 at start & end
                starts = np.flatnonzero(diff == 1)  # Transition from 0 → 1
                ends = np.flatnonzero(diff == -1)  # Transition from 1 → 0

                # Store results using list comprehension
                late_fusion_av.extend(
                    {'event_label': event_label, 'start': int(s), 'end': int(e)}
                    for s, e in zip(starts, ends)
                )

            return {video_id: late_fusion_av}

    def filter_classes(self, labels, video_transformed, audio_transformed):
        similarities = self.model(labels, video_transformed, audio_transformed, similarity_type='combined', vision_mode='video')

        combined_events = self.events_above_threshold(labels, similarities['combined'], self.filter_threshold)
        video_events = self.events_above_threshold(labels, similarities['video'], self.filter_threshold)
        audio_events = self.events_above_threshold(labels, similarities['audio'], self.filter_threshold)

        return combined_events, video_events, audio_events

    def set_thresholds(self, similarities, labels, embeddings):
        thresholds = np.full((similarities.shape[0], len(labels)), self.threshold_stage1)
        count_events = np.zeros(len(labels))

        for event_dim in range(similarities.shape[0] - 1):
            tensor_slice_np = similarities[event_dim].cpu().numpy()
            indices = np.where(tensor_slice_np > thresholds[event_dim])[0]
            count_events[indices] += 1
            offset = (self.threshold_stage1 * np.e**(-self.gamma*count_events))
            if "cosine" in self.method and embeddings is not None:
                offset *= cosine_similarity(embeddings[event_dim].unsqueeze(0), embeddings[event_dim+1].unsqueeze(0)).item()

            if "bbse" in self.method:
                label_shift_ratio = estimate_labelshift_ratio((similarities[:event_dim+1].cpu().numpy() > thresholds[:event_dim+1])*1, similarities[:event_dim+1].cpu().numpy(), 
                                                                np.expand_dims(similarities[event_dim + 1].cpu().numpy(), 0), len(labels))
                offset *= label_shift_ratio

            thresholds[event_dim + 1] = thresholds[event_dim] - offset
        
        return thresholds

    def refine_segments(self, video_events, decord_vr, waveform_and_sr, labels, similarity_type):
        if (len(video_events) == 1 and len(next(iter(video_events.values()))) == 1) or (not video_events):
            return video_events
        else:
            refined_segments = {}
            past_segments = {}
            for event, time_ranges in video_events.items():
                for start_time, end_time in time_ranges:
                    audio_transformed = self.audio_transforms(waveform_and_sr, start_sec=start_time, end_sec=end_time).to(self.device)
                    video_transformed = self.vision_transforms(decord_vr, transform_type='video', start=start_time, end=end_time).to(self.device)

                    similarities = self.model(labels, video_transformed, audio_transformed, similarity_type=similarity_type, vision_mode='video')[similarity_type]

                    if self.dataset == "AVE":
                        events = [labels[similarities.argmax().item()]]
                    else:    
                        events = self.events_above_threshold(labels, similarities, self.threshold_stage2)
                    similarities = similarities[0]

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
        
    def optimize(self, similarities, decord_vr, waveform_and_sr, labels, video_id, similarity_type, embeddings):
        image_events_dict = {}
        thresholds = np.full((similarities.shape[0], len(labels)), self.threshold_stage1) if self.method == 'naive' \
                    else self.set_thresholds(similarities, labels, embeddings)
        
        for event_dim in range(similarities.shape[0]):
            tensor_slice_np = similarities[event_dim].cpu().numpy()
            indices = np.where(tensor_slice_np > thresholds[event_dim])[0]
            events = [labels[i] for i in indices]
            image_events_dict[f"frame-{event_dim}"] = events

        results = self.optimize_video_events(image_events_dict)
        results = self.increment_end_times(results)
        results = self.merge_consecutive_segments(results)
        results = self.filter_events(results)
        
        if not self.without_refine_segments:
            results = self.refine_segments(results, decord_vr, waveform_and_sr, labels, similarity_type)
        
        return self.convert_results(results, video_id)

    def __call__(self, labels, decord_vr, waveform_and_sr, video_id):

        audio_transformed = self.audio_transforms(waveform_and_sr).to(self.device)
        combined_labels, video_labels, audio_labels = labels, labels, labels
        
        if not self.without_filter_classes:
            video_transformed = self.vision_transforms(decord_vr, transform_type='video').to(self.device)
            combined_labels, video_labels, audio_labels = self.filter_classes(labels, video_transformed, audio_transformed)
        
        video_transformed = self.vision_transforms(decord_vr, transform_type='image').to(self.device)
        video_similarties = self.model(video_labels, video_transformed, audio_transformed, similarity_type='image', vision_mode='image')
        audio_similarites = self.model(audio_labels, video_transformed, audio_transformed, similarity_type='audio')

        video_results = self.optimize(video_similarties['image'], decord_vr, waveform_and_sr, video_labels, video_id, "video", video_similarties['image_features'])
        audio_results = self.optimize(audio_similarites['audio'].repeat(video_similarties['image'].shape[0], 1), decord_vr, waveform_and_sr, audio_labels, video_id, "audio", None)

        if self.fusion == "early":
            combined_similarities = self.model(combined_labels, video_transformed, audio_transformed, similarity_type='combined', vision_mode='image')
            combined_results = self.optimize(combined_similarities['combined'], decord_vr, waveform_and_sr, combined_labels, video_id, "combined", combined_similarities['image_features'])
        elif self.fusion == "late":
           combined_results = self.late_fusion(video_id, video_results, audio_results)

        return combined_results, video_results, audio_results