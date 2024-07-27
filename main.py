from PIL import Image
from models.imagebindmodel.imagebind.data import load_and_transform_text, load_and_transform_audio_data
import torch
from models.imagebindmodel.imagebind.models import imagebind_model
from models.imagebindmodel.imagebind.models.imagebind_model import ModalityType
from dataset import AVE
import argparse
import numpy as np
from data_transforms import image_transforms_imagebind

def predict(labels, frames, audio_file_name):
    modality_inputs = {
        ModalityType.TEXT: load_and_transform_text(labels, device),
        ModalityType.VISION: frames.to(device),
        # ModalityType.AUDIO: load_and_transform_audio_data([audio_file], device),
    }

    with torch.no_grad():
        embeddings = model(modality_inputs)
        video_text_similarity = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
        # audio_text_similarity = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1)

    image_events_dict = {}      
    for event_dim in range(video_text_similarity.shape[0]):
        tensor_slice_np = video_text_similarity[event_dim].cpu().numpy()
        indices = np.where(tensor_slice_np > threshold)[0]
        events = [labels[i] for i in indices]
        # values = tensor_slice_np[indices]
        image_events_dict[f"frame-{event_dim}"] = events


    single_video_events = optimize_video_events(image_events_dict)
    print(single_video_events)


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', required=True, type=str)
    parser.add_argument('--annotations_file_path', required=True, type=str)
    parser.add_argument('--gpu_id', default=-1, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)

    args = parser.parse_args()

    threshold = args.threshold

    dataset = AVE(args.video_dir_path, #extract those to the argparse
                  args.annotations_file_path,
                  frames_transforms=image_transforms_imagebind)
    
    labels = list(dataset.class2idx.keys())

    device = f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    for sample in dataset:
        frames, audio_file_name, label_dict = sample
        predict(labels, frames, audio_file_name)