import torch
import numpy as np
from models.languagebindmodel.languagebind import to_device
from data_transforms import VisionTransform, AudioTransform
from label_shift import estimate_labelshift_ratio
from models.languagebindmodel.languagebind import LanguageBind, LanguageBindImageTokenizer, to_device
import clip
import laion_clap


def pad_similarities(vision_text_similarity, audio_text_similarity, device):
    if vision_text_similarity.shape[0] < audio_text_similarity.shape[0]:
        pad = torch.zeros_like(vision_text_similarity[0].unsqueeze(0)).to(device)
        pad = pad.repeat_interleave(audio_text_similarity.shape[0] - vision_text_similarity.shape[0])
        vision_text_similarity = torch.cat((vision_text_similarity, pad), dim=0)
    elif vision_text_similarity.shape[0] > audio_text_similarity.shape[0]:
        pad = torch.zeros_like(audio_text_similarity[0].unsqueeze(0)).to(device)
        pad = pad.repeat(vision_text_similarity.shape[0] - audio_text_similarity.shape[0], 1)
        audio_text_similarity = torch.cat((audio_text_similarity, pad), dim=0)
    
    return vision_text_similarity, audio_text_similarity

def norm_similarities(similarities):
        similarities = (similarities - torch.mean(similarities, dim=-1, keepdim=True)) / torch.std(similarities, dim=-1, keepdim=True)
        similarities = torch.sigmoid(similarities)
        return similarities

def calculate_combined_similarity(video_text_similarity, audio_text_similarity, alpha):
    return (1 - alpha) * video_text_similarity + alpha * audio_text_similarity


class LanguageBind_model:
    audio_frames = 10

    def __init__(self, device, alpha):
        clip_type = {
            'video': 'LanguageBind_Video_FT', 
            'audio': 'LanguageBind_Audio_FT',
            'image': 'LanguageBind_Image',
        }
        self.device = device
        self.model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
        self.model = self.model.to(self.device)
        self.model.eval()
        pretrained_ckpt = f'lb203/LanguageBind_Image'
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
        self.vision_transforms = VisionTransform(model="LanguageBind")
        self.audio_transforms = AudioTransform(model="LanguageBind")
        self.alpha = alpha


    def __call__(self, labels, vision_transformed, audio_transformed, similarity_type='audio', vision_mode='video'):
        """
        Compute and return similarity scores based on the requested modality.

        similarity_type options:
        - 'audio': returns only audio-text similarity
        - 'image': returns only image-text similarity
        - 'video': returns only video-text similarity
        - 'combined': returns a weighted combination of audio and vision similarity (requires audio)
        
        vision_mode options (only relevant for 'combined'):
        - 'video': use video-text similarity for combination
        - 'image': use image-text similarity for combination
        """
        preprocessed_labels = [f"A {label.replace('_', ' ').lower()}" for label in labels]

        languagebind_inputs = {'language': self.tokenizer(preprocessed_labels, max_length=77, padding='max_length',
                                                          truncation=True, return_tensors='pt').to(self.device)}
        
        if similarity_type == 'audio':
            assert audio_transformed is not None, "Audio input is required for audio similarity."
            languagebind_inputs['audio'] = {"pixel_values": audio_transformed}
        
        elif similarity_type in ['image', 'video']:
            assert vision_transformed is not None, f"{similarity_type.capitalize()} input is required."
            languagebind_inputs[similarity_type] = {"pixel_values": vision_transformed}

        elif similarity_type == 'combined':
            assert audio_transformed is not None and vision_transformed is not None, \
                "Both audio and vision inputs are required for combined similarity."
            languagebind_inputs.update({'audio': {"pixel_values": audio_transformed},
                                        vision_mode: {"pixel_values": vision_transformed}})

        else:
            raise ValueError(f"Invalid similarity_type: {similarity_type}")

        with torch.no_grad():
            embeddings = self.model(languagebind_inputs)

        similarities = {}

        # Compute audio similarity
        if 'audio' in embeddings:
            similarities['audio'] = norm_similarities(embeddings['audio'] @ embeddings['language'].T)

        # Compute vision similarity (image or video)
        if 'image' in embeddings:
            similarities['image'] = norm_similarities(embeddings['image'] @ embeddings['language'].T)
            similarities['image_features'] = embeddings['image']
        
        if 'video' in embeddings:
            similarities['video'] = norm_similarities(embeddings['video'] @ embeddings['language'].T)

        # Compute combined similarity with flexible vision choice
        if similarity_type == 'combined':
            assert 'audio' in similarities, "Audio similarity is mandatory for combined similarity."
            assert vision_mode in similarities, f"Vision mode '{vision_mode}' is missing in similarity computations."

            # Pad similarities if needed
            similarities['audio'] = similarities['audio'].repeat(similarities[vision_mode].shape[0], 1)
            vision_text_similarity, audio_text_similarity = pad_similarities(similarities[vision_mode], similarities['audio'], self.device)

            similarities['combined'] = calculate_combined_similarity(vision_text_similarity, audio_text_similarity, self.alpha)

        return similarities


class CLIP_CLAP_model:
    def __init__(self, device):
        self.device = device
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.vision_transforms = VisionTransform(model="CLIP")
        self.audio_transforms = AudioTransform(model="CLAP")

        self.clap = laion_clap.CLAP_Module(enable_fusion=False).to(self.device)
        self.clap.load_ckpt() # download the default pretrained checkpoint.
    
    def __call__(self, labels, video_transformed, audio_trnasformed, similarity_type):
        clap_text_labels = [f"This is a sound of {label}" for label in labels]
        clip_text_labels = clip.tokenize([f"a {label}" for label in labels]).to(self.device)
        
        with torch.no_grad():
            audio_features = self.clap.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
            clap_text_features = self.clap.get_text_embedding(clap_text_labels, use_tensor=True).to(self.device)

            image_features = self.clip_model.encode_image(images)
            clip_text_features = self.clip_model.encode_text(clip_text_labels)

        
        if similarity_type == "image":
            # if image_features.shape[0] != audio_features.shape[0]:
            #     audio_features = audio_features.repeat_interleave(self.sample_audio_sec, dim=0)
            vision_text_similarity = image_features @ clip_text_features.T
        
        elif similarity_type == "video":
            vision_text_similarity = image_features @ clip_text_features.T
            vision_text_similarity = vision_text_similarity.mean(dim=0).unsqueeze(0)
        else:
            raise ValueError("Similarity type is wrong !")

        audio_text_similarity = audio_features @ clap_text_features.T
            
        pad_similarities(vision_text_similarity, audio_text_similarity, self.device)

        return norm_similarities(vision_text_similarity), norm_similarities(audio_text_similarity), image_features

        