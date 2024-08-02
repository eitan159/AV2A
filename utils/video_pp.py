import contextlib
import io
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2

def suppress_output(func):
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)
    return wrapper

@suppress_output
def crop_video_and_extract_audio(video_path, start_time, end_time, output_video_path, output_audio_path):
    video = VideoFileClip(video_path)
    cropped_video = video.subclip(start_time, end_time)
    cropped_video.write_videofile(output_video_path, codec='libx264')
    audio = cropped_video.audio
    audio.write_audiofile(output_audio_path, codec='pcm_s16le')

@suppress_output
def get_video_duration(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)      # Get frames per second
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)  # Get total frames
    duration = frame_count / fps
    return duration

@suppress_output
def extract_audio(video_path, audio_output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_output_path)
    video_clip.close()
    audio_clip.close()
