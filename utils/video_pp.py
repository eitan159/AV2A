import contextlib
import io
from moviepy.video.io.VideoFileClip import VideoFileClip


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
    