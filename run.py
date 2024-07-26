import torch
import numpy as np
import math
from models.languagebindmodel.languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from utils.video_pp import crop_video, extract_audio_from_video, get_video_length


# Set up device and constants
device = torch.device('cuda:1')
clip_type = {
    'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
    # 'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
}
PATH_FOR_SAVING_CROPED_VIDS = '/home/shaulov/work/zeroshot_AVE/cropped_video.mp4'

# Load model and tokenizer
model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
model = model.to(device)
model.eval()
pretrained_ckpt = f'lb203/LanguageBind_Image'
tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}


def run(video_paths, labels):
    for video_path in video_paths:
        event, max_score = get_single_video_event(video_path, labels)
        optimize_localization(video_path, event, labels, max_score, 0, 10)

def get_single_video_event(video_path, labels):
    audio_path = video_path.replace(".mp4", "-audio.mp4")
    extract_audio_from_video(video_path, audio_path)

    inputs = {
        'video': to_device(modality_transform['video'](video_path), device),
        # 'audio': to_device(modality_transform['audio'](audio_path), device),
    }
    inputs['language'] = to_device(tokenizer(labels, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)

    with torch.no_grad():
        embeddings = model(inputs)

    video_text_similarity = torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy()
    # audio_text_similarity = torch.softmax(embeddings['audio'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy()
    audio_text_similarity = 0

    combined_similarity = video_text_similarity + audio_text_similarity
    max_index = np.argmax(combined_similarity)
    max_score = combined_similarity[0][max_index]

    return labels[max_index], max_score


def optimize_localization(video_path, event, labels, max_score, event_st, event_et):
    video_len = round(get_video_length(video_path))
    step_size = math.ceil(0.2 * video_len)
    optimized_event_st = optimize_time(event, event_st, event_et, video_len, step_size, max_score, labels, video_path, is_start=True)
    optimized_event_et = optimize_time(event, event_st, event_et, video_len, step_size, max_score, labels, video_path, is_start=False)
    print(optimized_event_st, optimized_event_et)

def optimize_time(event, event_st, event_et, video_len, step_size, max_score, labels, video_path, is_start=True):
    while True:
        previous_max_score = max_score

        #  optimize the boundery to the right side
        if is_start:
            new_time = adjust_time(event_st, step_size, video_len, True)
            crop_path = crop_video(video_path, new_time, event_et, PATH_FOR_SAVING_CROPED_VIDS)
        else:
            new_time = adjust_time(event_et, step_size, video_len, True)
            crop_path = crop_video(video_path, event_st, new_time, PATH_FOR_SAVING_CROPED_VIDS)

        crop_event, event_score = get_single_video_event(crop_path, labels)

        if crop_event == event and event_score > max_score:
            max_score = event_score
            if is_start:
                event_st = new_time
            else:
                event_et = new_time

        # optimize the boundery to the left side
        if is_start:
            new_time = adjust_time(event_st, step_size, video_len, False)
            crop_path = crop_video(video_path, new_time, event_et, PATH_FOR_SAVING_CROPED_VIDS)
        else:
            new_time = adjust_time(event_et, step_size, video_len, False)
            crop_path = crop_video(video_path, event_st, new_time, PATH_FOR_SAVING_CROPED_VIDS)

        crop_event, event_score = get_single_video_event(crop_path, labels)

        if crop_event == event and event_score > max_score:
            max_score = event_score
            if is_start:
                event_st = new_time
            else:
                event_et = new_time

        if step_size == 1 and previous_max_score == max_score:
            break
        elif previous_max_score < max_score:
            continue
        else:
            step_size = math.ceil(step_size / 2)

    return event_st if is_start else event_et


def adjust_time(time, step_size, video_len, compare_to_start):
    if compare_to_start:
        return max(0, time - step_size) if time - step_size > 0 else time
    else:
        return min(video_len, time + step_size) if time + step_size < video_len else time


if __name__ == '__main__':
    video_paths = ['/home/shaulov/work/zeroshot_AVE/MH3m4AwEcRY.mp4']
    # video_paths = ['/home/shaulov/work/zeroshot_AVE/eRF4KAXdn0w.mp4']

    labels = ["Bark.", 'Church bell.']
    run(video_paths, labels)

