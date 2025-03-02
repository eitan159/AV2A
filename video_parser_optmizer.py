import torch
import numpy as np
from models.languagebindmodel.languagebind import to_device
from data_transforms import VisionTransform, AudioTransform
from label_shift import estimate_labelshift_ratio
from models.languagebindmodel.languagebind import LanguageBind, LanguageBindImageTokenizer, to_device
from model import LanguageBind_model
import clip
import laion_clap

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

        self.model = LanguageBind_model(device, alpha) if backbone == 'language_bind' else None
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
            if "cosine" in self.method:

                vector1 = embeddings[event_dim].cpu().numpy()
                vector2 = embeddings[event_dim+1].cpu().numpy()
                cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                cosine_similarity = np.clip(cosine_similarity, 0, 1)
                offset *= cosine_similarity

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

                    similarities = self.model(labels, audio_transformed, video_transformed, similarity_type=similarity_type, vision_mode='video')[similarity_type]

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

        audio_transformed = self.audio_transforms(waveform_and_sr).unsqueeze(0).to(self.device)
        combined_labels, video_labels, audio_labels = labels, labels, labels
        
        if not self.without_filter_classes:
            video_transformed = self.vision_transforms(decord_vr, transform_type='video').unsqueeze(0).to(self.device)
            combined_labels, video_labels, audio_labels = self.filter_classes(labels, video_transformed, audio_transformed)
        
        video_transformed = self.vision_transforms(decord_vr, transform_type='image').to(self.device)
        video_similarties = self.model(video_labels, video_transformed, audio_transformed, similarity_type='image')
        audio_similarites = self.model(audio_labels, video_transformed, audio_transformed, similarity_type='audio')

        video_results = self.optimize(video_similarties['image'], decord_vr, waveform_and_sr, video_labels, video_id, "video", video_similarties['image_features'])
        audio_results = self.optimize(audio_similarites['audio'].repeat(video_similarties['image'].shape[0], 1), decord_vr, waveform_and_sr, audio_labels, video_id, "audio", None)

        if self.fusion == "early":
            combined_similarities = self.model(combined_labels, video_transformed, audio_transformed, similarity_type='combined', vision_mode='image')
            combined_results = self.optimize(combined_similarities['combined'], decord_vr, waveform_and_sr, combined_labels, video_id, "combined", combined_similarities['image_features'])

        return combined_results, video_results, audio_results
        
    

class VideoParserOptimizer_LanguageBind(VideoParserOptimizer):
    def __init__(self, method, labels, device, sample_audio_sec, alpha,
                 filter_threshold, threshold_stage1, threshold_stage2, gamma, without_filter_classes,
                 without_refine_segments, dataset, labels_shift_iters) -> None:
        
        super().__init__(method, labels, device, sample_audio_sec, alpha,
                        filter_threshold, threshold_stage1, threshold_stage2, gamma, without_filter_classes,
                        without_refine_segments, dataset, labels_shift_iters)
        
        clip_type = {
            'video': 'LanguageBind_Video_FT', 
            'audio': 'LanguageBind_Audio_FT',
            'image': 'LanguageBind_Image',
        }

        self.model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
        self.model = self.model.to(device)
        self.model.eval()
        pretrained_ckpt = f'lb203/LanguageBind_Image'
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
        self.vision_transforms = VisionTransform()
        self.audio_transforms = AudioTransform()
    
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
    
    def predict(self, labels, decord_vr, waveform_and_sr, video_id, fusion):
        if not self.without_filter_classes:
            combined_filtered_labels, video_filtered_labels, audio_filtered_labels = self.filter_classes(labels, decord_vr, waveform_and_sr, self.filter_threshold)
        else:
            combined_filtered_labels, video_filtered_labels, audio_filtered_labels = labels, labels, labels
        
        inputs = {
            'image': {"pixel_values": self.vision_transforms(decord_vr, transform_type='image').to(self.device)},
            'audio': {"pixel_values": self.audio_transforms.split_sample_audio(waveform_and_sr, self.sample_audio_sec, FM_name="language_bind").to(self.device)},
        }
        if fusion == "early":
            combined_similarities, _, _, embeddings_combined = self.get_similiraties(combined_filtered_labels, inputs, self.alpha)
        
        _, video_text_similarties, _, embeddings_video_text = self.get_similiraties(video_filtered_labels, inputs, self.alpha)
        _, _, audio_text_similarites, embeddings_audio_text = self.get_similiraties(audio_filtered_labels, inputs, self.alpha)

        video_results = self.optimize(video_text_similarties, decord_vr, waveform_and_sr, video_filtered_labels, video_id, "video", embeddings_video_text)
        audio_results = self.optimize(audio_text_similarites, decord_vr, waveform_and_sr, audio_filtered_labels, video_id, "audio", embeddings_audio_text)
        
        if fusion == "early":
            combined_results = self.optimize(combined_similarities, decord_vr, waveform_and_sr, combined_filtered_labels, video_id, "combined", embeddings_combined)
        else:
            late_fusion_av = []
            for video_result in video_results[video_id]:
                for audio_result in audio_results[video_id]:
                    if video_result['event_label'] == audio_result['event_label']:
                        pred_audio = np.asarray([1 if audio_result['start'] <= i < audio_result['end'] else 0 for i in range(10)])
                        pred_video = np.asarray([1 if video_result['start'] <= i < video_result['end'] else 0 for i in range(10)])

                        pred_av = pred_audio * pred_video

                        diff = np.diff(pred_av, prepend=0, append=0)
    
                        # Find the start and end indices of segments with 1s
                        starts = np.where(diff == 1)[0]
                        ends = np.where(diff == -1)[0] - 1
                        
                        # Combine the start and end indices into segments
                        segments = [[start, end + 1] for start, end in zip(starts, ends)]
                        for seg in segments:
                            late_fusion_av.append({
                                'event_label': video_result['event_label'],
                                'start': int(seg[0]),
                                'end': int(seg[1]),
                                }
                            )
            
            combined_results = {video_id: late_fusion_av}

        return combined_results, video_results, audio_results

    def optimize(self, similarities, decord_vr, waveform_and_sr, labels, video_id, similarity_type, embeddings):
        image_events_dict = {}
        thresholds = np.full((similarities.shape[0], len(labels)), self.threshold_stage1)
        count_events = np.zeros(len(labels))
        for _ in range(self.labels_shift_iters):
            for event_dim in range(similarities.shape[0] - 1):
                tensor_slice_np = similarities[event_dim].cpu().numpy()
                indices = np.where(tensor_slice_np > thresholds[event_dim])[0]
                events = [labels[i] for i in indices]

                if self.method == "cosine":
                    count_events[indices] += 1

                    vector1 = embeddings['image'][event_dim].cpu().numpy()
                    vector2 = embeddings['image'][event_dim+1].cpu().numpy()

                    cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                    cosine_similarity = np.clip(cosine_similarity, 0, 1)

                    offset = (self.threshold_stage1 * np.e**(-self.gamma*count_events)) * cosine_similarity
                    thresholds[event_dim + 1] = thresholds[event_dim] - offset

                elif "bbse" in self.method:
                    count_events[indices] += 1
                    label_shift_ratio = estimate_labelshift_ratio((similarities[:event_dim+1].cpu().numpy() > thresholds[:event_dim+1])*1, similarities[:event_dim+1].cpu().numpy(), 
                                                                    np.expand_dims(similarities[event_dim + 1].cpu().numpy(), 0), len(labels))

                    offset = (self.threshold_stage1 * np.e**(-self.gamma*count_events)) * label_shift_ratio

                    if self.method == 'bbse-cosine':
                        vector1 = embeddings['image'][event_dim].cpu().numpy()
                        vector2 = embeddings['image'][event_dim+1].cpu().numpy()

                        cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                        cosine_similarity = np.clip(cosine_similarity, 0, 1)
                        
                        offset *= cosine_similarity

                    thresholds[event_dim + 1] = thresholds[event_dim] - offset
        
        for event_dim in range(similarities.shape[0]):
            tensor_slice_np = similarities[event_dim].cpu().numpy()
            indices = np.where(tensor_slice_np > thresholds[event_dim])[0]
            events = [labels[i] for i in indices]
            image_events_dict[f"frame-{event_dim}"] = events

        # tensor_slice_np = similarities[len(similarities) - 1].cpu().numpy()
        # indices = np.where(tensor_slice_np > thresholds[len(similarities) - 1])[0]
        # events = [labels[i] for i in indices]
        # image_events_dict[f"frame-{len(similarities) - 1}"] = events


        results = self.optimize_video_events(image_events_dict)
        results = self.increment_end_times(results)
        results = self.merge_consecutive_segments(results)
        results = self.filter_events(results)
        
        if not self.without_refine_segments:
            results = self.refine_segments(results, decord_vr, waveform_and_sr, labels, similarity_type)
        
        return self.convert_results(results, video_id)
    
class VideoParserOptimizer_CLIP_CLAP(VideoParserOptimizer):
    def __init__(self, method, labels, device, sample_audio_sec, alpha,
                 filter_threshold, threshold_stage1, threshold_stage2, gamma, without_filter_classes,
                 without_refine_segments, dataset, labels_shift_iters) -> None:
        
        super().__init__(method, labels, device, sample_audio_sec, alpha,
                        filter_threshold, threshold_stage1, threshold_stage2, gamma, without_filter_classes,
                        without_refine_segments, dataset, labels_shift_iters)

        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.vision_transforms = VisionTransform()
        self.audio_transforms = AudioTransform()

        self.clap = laion_clap.CLAP_Module(enable_fusion=False).to(self.device)
        self.clap.load_ckpt() # download the default pretrained checkpoint.
    
    def predict(self, labels, decord_vr, waveform_and_sr, video_id, fusion):

        images = self.vision_transforms(decord_vr, transform_type='image_clip').to(self.device)
        audio_data = torch.as_tensor(waveform_and_sr[0]).to(self.device)

        if not self.without_filter_classes:
            combined_filtered_labels, video_filtered_labels, audio_filtered_labels = self.filter_classes(images, audio_data)
        else:
            combined_filtered_labels, video_filtered_labels, audio_filtered_labels = labels, labels, labels
        
        audio_data = self.audio_transforms.split_sample_audio(waveform_and_sr, self.sample_audio_sec, FM_name="CLAP")
        
        combined_similarities, _, _, embeddings = self.get_similiraties(combined_filtered_labels, images, audio_data, similarity_type='image')
        _, video_text_similarties, _, _ = self.get_similiraties(video_filtered_labels, images, audio_data, similarity_type='image')
        _, _, audio_text_similarites, _ = self.get_similiraties(audio_filtered_labels, images, audio_data, similarity_type='image')

        video_results = self.optimize(video_text_similarties, images, waveform_and_sr, video_filtered_labels, video_id, "video", embeddings)
        audio_results = self.optimize(audio_text_similarites, images, waveform_and_sr, audio_filtered_labels, video_id, "audio", embeddings)

        if fusion == "early":
            combined_results = self.optimize(combined_similarities, images, waveform_and_sr, combined_filtered_labels, video_id, "combined", embeddings)
        else:
            late_fusion_av = []
            for video_result in video_results[video_id]:
                for audio_result in audio_results[video_id]:
                    if video_result['event_label'] == audio_result['event_label']:
                        pred_audio = np.asarray([1 if audio_result['start'] <= i < audio_result['end'] else 0 for i in range(10)])
                        pred_video = np.asarray([1 if video_result['start'] <= i < video_result['end'] else 0 for i in range(10)])

                        pred_av = pred_audio * pred_video

                        diff = np.diff(pred_av, prepend=0, append=0)
    
                        # Find the start and end indices of segments with 1s
                        starts = np.where(diff == 1)[0]
                        ends = np.where(diff == -1)[0] - 1
                        
                        # Combine the start and end indices into segments
                        segments = [[start, end + 1] for start, end in zip(starts, ends)]
                        for seg in segments:
                            late_fusion_av.append({
                                'event_label': video_result['event_label'],
                                'start': int(seg[0]),
                                'end': int(seg[1]),
                                }
                            )
            
            combined_results = {video_id: late_fusion_av}

        return combined_results, video_results, audio_results

    def filter_classes(self, images, audio_data):
        combined_similarities, video_text_similarity, audio_text_similarity, _ = self.get_similiraties(self.labels, images, audio_data, similarity_type='video')

        combined_events = self.events_above_threshold(self.labels, combined_similarities, self.filter_threshold)
        video_events = self.events_above_threshold(self.labels, video_text_similarity, self.filter_threshold)
        audio_events = self.events_above_threshold(self.labels, audio_text_similarity, self.filter_threshold)

        return combined_events, video_events, audio_events

    def get_similiraties(self, labels, images, audio_data, similarity_type):

        clap_text_labels = [f"This is a sound of {t}" for t in labels]
        clip_text_labels = clip.tokenize([f"a {t}" for t in labels]).to(self.device)
        
        with torch.no_grad():
            audio_features = self.clap.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
            clap_text_features = self.clap.get_text_embedding(clap_text_labels, use_tensor=True).to(self.device)

            image_features = self.clip_model.encode_image(images)
            clip_text_features = self.clip_model.encode_text(clip_text_labels)

        
        if similarity_type == "image":
            if image_features.shape[0] != audio_features.shape[0]:
                audio_features = audio_features.repeat_interleave(self.sample_audio_sec, dim=0)
                video_text_similarity = image_features @ clip_text_features.T
        
        elif similarity_type == "video":
            video_text_similarity = image_features @ clip_text_features.T
            video_text_similarity = video_text_similarity.mean(dim=0).unsqueeze(0)
        else:
            raise ValueError("Similarity type is wrong !")

        audio_text_similarity = audio_features @ clap_text_features.T
            
        if video_text_similarity.shape[0] < audio_text_similarity.shape[0]:
            pad = torch.zeros_like(video_text_similarity[0].unsqueeze(0)).to(self.device)
            pad = pad.repeat_interleave(audio_text_similarity.shape[0] - video_text_similarity.shape[0])
            video_text_similarity = torch.cat((video_text_similarity, pad), dim=0)
        elif video_text_similarity.shape[0] > audio_text_similarity.shape[0]:
            pad = torch.zeros_like(audio_text_similarity[0].unsqueeze(0)).to(self.device)
            pad = pad.repeat(video_text_similarity.shape[0] - audio_text_similarity.shape[0], 1)
            audio_text_similarity = torch.cat((audio_text_similarity, pad), dim=0)
        
        combined_similarities, video_text_similarity_sigmoid_norm, audio_text_similarity_sigmoid_norm = self.proccess_similarities(video_text_similarity, audio_text_similarity)

        return combined_similarities, video_text_similarity_sigmoid_norm, audio_text_similarity_sigmoid_norm, image_features


    def optimize(self, similarities, images, waveform_and_sr, labels, video_id, similarity_type, embeddings):
        image_events_dict = {}
        thresholds = np.full((similarities.shape[0], len(labels)), self.threshold_stage1)
        count_events = np.zeros(len(labels))
        for _ in range(self.labels_shift_iters):
            for event_dim in range(similarities.shape[0] - 1):
                tensor_slice_np = similarities[event_dim].cpu().numpy()
                indices = np.where(tensor_slice_np > thresholds[event_dim])[0]
                events = [labels[i] for i in indices]

                if self.method == "cosine":
                    count_events[indices] += 1

                    vector1 = embeddings[event_dim].cpu().numpy()
                    vector2 = embeddings[event_dim+1].cpu().numpy()

                    cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                    cosine_similarity = np.clip(cosine_similarity, 0, 1)

                    offset = (self.threshold_stage1 * np.e**(-self.gamma*count_events)) * cosine_similarity
                    thresholds[event_dim + 1] = thresholds[event_dim] - offset

                elif "bbse" in self.method:
                    count_events[indices] += 1
                    label_shift_ratio = estimate_labelshift_ratio((similarities[:event_dim+1].cpu().numpy() > thresholds[:event_dim+1])*1, similarities[:event_dim+1].cpu().numpy(), 
                                                                    np.expand_dims(similarities[event_dim + 1].cpu().numpy(), 0), len(labels))

                    offset = (self.threshold_stage1 * np.e**(-self.gamma*count_events)) * label_shift_ratio

                    if self.method == 'bbse-cosine':
                        vector1 = embeddings[event_dim].cpu().numpy()
                        vector2 = embeddings[event_dim+1].cpu().numpy()

                        cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                        cosine_similarity = np.clip(cosine_similarity, 0, 1)
                        
                        offset *= cosine_similarity

                    thresholds[event_dim + 1] = thresholds[event_dim] - offset
        

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
            results = self.refine_segments(results, images, waveform_and_sr, labels, similarity_type)
        
        return self.convert_results(results, video_id)

    def refine_segments(self, video_events, images, waveform_and_sr, labels, similarity_type):
        if (len(video_events) == 1 and len(next(iter(video_events.values()))) == 1) or (not video_events):
            return video_events
        else:
            refined_segments = {}
            past_segments = {}
            for event, time_ranges in video_events.items():
                for start_time, end_time in time_ranges:

                    tmp_images = images[start_time: end_time, :, :, :]
                    audio_data = torch.as_tensor(self.audio_transforms.crop_audio(waveform_and_sr[0], waveform_and_sr[1], start_sec=start_time, end_sec=end_time)).to(self.device)

                    combined_similarities, video_text_similarties, audio_text_similarties, _ = self.get_similiraties(labels, tmp_images, audio_data, similarity_type='video')
                    
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