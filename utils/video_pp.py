import os
import sys
import contextlib
from moviepy.editor import VideoFileClip

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