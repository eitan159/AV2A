import os
import sys
import contextlib
import numpy as np
import torch
from models.imagebindmodel.imagebind import data
from models.imagebindmodel.imagebind.models import imagebind_model
from models.imagebindmodel.imagebind.models.imagebind_model import ModalityType
from utils.video_pp import convert_video_to_sorted_frames, extract_frames, get_frame_number

FPS = 1
FRAME_OUTPUT_DIR = "/home/shaulov/work/zeroshot_AVE/output_frames"
threshold = 0.5
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

def run(video_paths, labels):

    video_events_dict = {}
    for video_path in video_paths:
        sorted_frame_paths = convert_video_to_sorted_frames(video_path, FRAME_OUTPUT_DIR, FPS)

        with torch.no_grad():
            embeddings = model(get_inputs_for_image_bind(sorted_frame_paths, labels))

        video_text_similarity = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
        image_events_dict = {}      
        for event_dim in range(video_text_similarity.shape[0]):
            tensor_slice_np = video_text_similarity[event_dim].cpu().numpy()
            indices = np.where(tensor_slice_np > threshold)[0]
            events = [labels[i] for i in indices]
            # values = tensor_slice_np[indices]

            image_events_dict[f"frame-{event_dim}"] = events


        single_video_events = optimize_video_events(image_events_dict)
        video_events_dict[video_path] = single_video_events

        
    print(video_text_similarity)



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



def get_inputs_for_image_bind(sorted_frame_paths, labels):
    input =  {
        ModalityType.TEXT: data.load_and_transform_text(labels, device),
        ModalityType.VISION: data.load_and_transform_vision_data(sorted_frame_paths, device),
    }

    return input



if __name__ == '__main__':
    # video_paths = ['/home/shaulov/work/zeroshot_AVE/MH3m4AwEcRY.mp4']
    # video_paths = ['/home/shaulov/work/zeroshot_AVE/eRF4KAXdn0w.mp4']
    video_paths = ['/home/shaulov/work/zeroshot_AVE/Zlwu4AROYzg.mp4']
    labels = ['Church bell', 'Bark', 'Male speech,', 'Fixed-wing aircraft','Race car','Female speech', 'Violin', 'Flute','Ukulele', 'Frying','ariel']
    run(video_paths, labels)


