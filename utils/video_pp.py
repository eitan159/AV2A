import os
import re
import sys
import contextlib
from moviepy.editor import VideoFileClip
from PIL import Image

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

def extract_audio_from_video(video_path, path_to_save_audio):
    with suppress_stdout_stderr():
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            if audio:
                audio.write_audiofile(path_to_save_audio, codec='pcm_s16le')
            else:
                print(f"The video: {video_path} has no audio!")
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {e}")
        finally:
            video.close()

def get_video_length(video_path):
    with suppress_stdout_stderr():
        try:
            video = VideoFileClip(video_path)
            duration = video.duration
            video.close()
            return duration
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

def crop_video(video_path, start_time, end_time, output_path):
    with suppress_stdout_stderr():
        try:
            video = VideoFileClip(video_path)
            cropped_video = video.subclip(start_time, end_time)
            cropped_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
            video.close()
            cropped_video.close()
            print(f"Successfully cropped video to {output_path}")
            return output_path
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        

def get_frame_number(path):
    match = re.search(r'frame_(\d+)\.png', path)
    return int(match.group(1)) if match else float('inf')


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

def convert_video_to_sorted_frames(video_path, frame_output_dir, fps):
    extract_frames(video_path, frame_output_dir, fps)
    image_paths = []
    for image_filename in os.listdir(frame_output_dir):
        image_path = os.path.join(frame_output_dir, image_filename)
        image_paths.append(image_path)

    return sorted(image_paths, key=get_frame_number)
   