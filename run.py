import numpy as np
import torch
from models.imagebindmodel.imagebind import data
from models.imagebindmodel.imagebind.models import imagebind_model
from models.imagebindmodel.imagebind.models.imagebind_model import ModalityType

threshold = 0.5
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

def run(labels, frames, audio_file_name, modality_inputs):
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


# if __name__ == '__main__':
#     video_paths = ['/home/shaulov/work/zeroshot_AVE/Zlwu4AROYzg.mp4']
#     labels = ['Church bell', 'Bark', 'Male speech,', 'Fixed-wing aircraft','Race car','Female speech', 'Violin', 'Flute','Ukulele', 'Frying','ariel']
#     run(labels, frames, audio_file_name)


