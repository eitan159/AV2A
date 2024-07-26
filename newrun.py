import os
import sys
import contextlib
import re
from moviepy.editor import VideoFileClip
from PIL import Image
from models.imagebindmodel.imagebind import data
import torch
from models.imagebindmodel.imagebind.models import imagebind_model
from models.imagebindmodel.imagebind.models.imagebind_model import ModalityType


FPS = 1
FRAME_OUTPUT_DIR = "/home/shaulov/work/zeroshot_AVE/output_frames"


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

def run(video_paths, labels):
    for video_path in video_paths:

        # extract_frames(video_path,FRAME_OUTPUT_DIR, FPS)
        image_paths = []
        for image_filename in os.listdir(FRAME_OUTPUT_DIR):
            image_path = os.path.join(FRAME_OUTPUT_DIR, image_filename)
            image_paths.append(image_path)

        sorted_frame_paths = sorted(image_paths, key=get_frame_number)

        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(labels, device),
            ModalityType.VISION: data.load_and_transform_vision_data(sorted_frame_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        video_text_similarity = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
        print(video_text_similarity)




        
def get_frame_number(path):
    match = re.search(r'frame_(\d+)\.png', path)
    return int(match.group(1)) if match else float('inf')

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def extract_frames(video_path, output_dir, fps):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with suppress_stdout_stderr():
        try:
            video = VideoFileClip(video_path)
            duration = int(video.duration)
            for t in range(0, duration):
                frame_time = t * (1.0 / fps)
                frame = video.get_frame(frame_time)
                frame_path = os.path.join(output_dir, f"frame_{t}.png")
                frame_image = Image.fromarray(frame)
                frame_image.save(frame_path)
            video.close()
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")



if __name__ == '__main__':
    video_paths = ['/home/shaulov/work/zeroshot_AVE/MH3m4AwEcRY.mp4']
    # video_paths = ['/home/shaulov/work/zeroshot_AVE/eRF4KAXdn0w.mp4']
    labels = ['Church bell', 'Bark', 'Male speech,', 'Fixed-wing aircraft','Race car','Female speech', 'Violin', 'Flute','Ukulele', 'Frying']
    run(video_paths, labels)
    extract_frames("/home/shaulov/work/zeroshot_AVE/MH3m4AwEcRY.mp4", "output_frames", fps=1)


