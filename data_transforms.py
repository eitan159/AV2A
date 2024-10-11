from torchvision import transforms
from torchvision.transforms import Compose, Lambda
import torchaudio
import numpy as np
import torch
from torchvision.transforms._transforms_video import NormalizeVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms import InterpolationMode

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
language_bind_image_transform = Compose(
        [
            # transforms.ToTensor(),
            Lambda(lambda x: x / 255.0),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image
        ]
    )

BICUBIC = InterpolationMode.BICUBIC
clip_image_transforms = Compose(
        [
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            Lambda(lambda x: x / 255.0),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
)

language_bind_video_transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

class VisionTransform:
    def __init__(self, num_frames = 8, images_num = 10, video_seconds = 10):
        self.video_num_frames = num_frames
        self.images_num = images_num
        self.video_seconds = video_seconds

    def __call__(self, decord_vr, transform_type, start=None, end=None):
        fps = len(decord_vr) // 10
        frames_indicis = list(range(len(decord_vr)))
        if start != None and end != None:
            frames_indicis = frames_indicis[start * fps: end * fps]
        
        if transform_type == "video":
            frames_indicis = np.linspace(0, len(frames_indicis) - 1, self.video_num_frames, dtype=int)
        else:
            frames_indicis = np.linspace(0, len(frames_indicis) - 1, self.images_num, dtype=int)

        if start != None and end != None:
            frames_indicis = [frame_idx + start * fps for frame_idx in frames_indicis]

        video_data = decord_vr.get_batch(frames_indicis)
        
        if transform_type == "video": 
            video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            video_data = language_bind_video_transform(video_data)
        elif transform_type == "image":
            video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            video_data = language_bind_image_transform(video_data)
        elif transform_type == "image_clip":
            video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            video_data = clip_image_transforms(video_data)
        else:
            raise ValueError("transform_type is not defined !!!")
        
        return video_data



DEFAULT_AUDIO_FRAME_SHIFT_MS = 10
class AudioTransform:
    def __init__(self):
        self.sample_rate = 16000
        self.num_mel_bins = 112
        self.target_length = 1036
        self.audio_mean = -4.2677393
        self.audio_std = 4.5689974
        self.mean = []
        self.std = []
        # mean=-4.2677393
        # std=4.5689974
        # self.norm = transforms.Normalize(mean=self.audio_mean, std=self.audio_std)


    def __call__(self, audio_data_and_origin_sr, start_sec=None, end_sec=None):
        audio_data, origin_sr = audio_data_and_origin_sr

        if start_sec != None and end_sec != None:
            audio_data = self.crop_audio(audio_data, origin_sr, start_sec, end_sec)

        if self.sample_rate != origin_sr:
            # print(audio_data.shape, origin_sr)
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=origin_sr, new_freq=self.sample_rate)
        waveform_melspec = self.waveform2melspec(audio_data)
        return waveform_melspec
    
    def crop_audio(self, waveform, sample_rate, start_sec, end_sec):
        # Calculate the start and end sample indices
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        
        # Ensure end_sample does not exceed the length of the waveform
        end_sample = min(end_sample, waveform.shape[1])

        # Crop the waveform
        cropped_waveform = waveform[:, start_sample:end_sample]
        
        return cropped_waveform

    def split_sample_audio(self, audio_data_and_origin_sr, sample_audio_sec, FM_name):
        origin_audio_data, origin_sr = audio_data_and_origin_sr
        total_seconds = origin_audio_data.shape[1] // origin_sr
        output = []
        for t in range(0, total_seconds, sample_audio_sec):
            audio_data = self.crop_audio(origin_audio_data, origin_sr, t, t + sample_audio_sec)
            if FM_name == "language_bind":
                output.append(self((audio_data, origin_sr)))
            else:
                output.append(torch.as_tensor(audio_data).squeeze(0))

        return torch.stack(output) if FM_name == "language_bind" else output


    def waveform2melspec(self, audio_data):
        mel = self.get_mel(audio_data)
        if mel.shape[0] > self.target_length:
            # split to three parts
            chunk_frames = self.target_length
            total_frames = mel.shape[0]
            ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
            # print('total_frames-chunk_frames:', total_frames-chunk_frames,
            #       'len(audio_data):', len(audio_data),
            #       'chunk_frames:', chunk_frames,
            #       'total_frames:', total_frames)
            if len(ranges[1]) == 0:  # if the audio is too short, we just use the first chunk
                ranges[1] = [0]
            if len(ranges[2]) == 0:  # if the audio is too short, we just use the first chunk
                ranges[2] = [0]
            # randomly choose index for each part
            idx_front = np.random.choice(ranges[0])
            idx_middle = np.random.choice(ranges[1])
            idx_back = np.random.choice(ranges[2])
            # idx_front = ranges[0][0]  # fixed
            # idx_middle = ranges[1][0]
            # idx_back = ranges[2][0]
            # select mel
            mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
            mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
            mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
            # print(total_frames, idx_front, idx_front + chunk_frames, idx_middle, idx_middle + chunk_frames, idx_back, idx_back + chunk_frames)
            # stack
            mel_fusion = torch.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
        elif mel.shape[0] < self.target_length:  # padding if too short
            n_repeat = int(self.target_length / mel.shape[0]) + 1
            # print(self.target_length, mel.shape[0], n_repeat)
            mel = mel.repeat(n_repeat, 1)[:self.target_length, :]
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        else:  # if equal
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        mel_fusion = mel_fusion.transpose(1, 2)  # [3, target_length, mel_bins] -> [3, mel_bins, target_length]

        # self.mean.append(mel_fusion.mean())
        # self.std.append(mel_fusion.std())
        mel_fusion = (mel_fusion - self.audio_mean) / (self.audio_std * 2)
        return mel_fusion

    def get_mel(self, audio_data):
        # mel shape: (n_mels, T)
        audio_data -= audio_data.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            audio_data,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
        )
        return mel  # (T, n_mels)


