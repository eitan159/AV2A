import wandb
from dataset import LLP
from models.languagebindmodel.languagebind import LanguageBind, LanguageBindImageTokenizer
from eval_metrics import calculate_metrices
from video_parser_optmizer import VideoParserOptimizer


def sweep():
    dataset = LLP(video_dir_path, audio_dir_path)
    
    labels = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']

    device = f"cuda:0"

    clip_type = {
        'video': 'LanguageBind_Video_FT', 
        'audio': 'LanguageBind_Audio_FT',
        'image': 'LanguageBind_Image',
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'lb203/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    # modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    model = VideoParserOptimizer(model, tokenizer, labels, device, config.sample_audio_sec, config.alpha, 
                            config.filter_threshold, config.threshold_stage1, config.threshold_stage2)

    combined_candidates, video_candidates, audio_candidates = [], [], []
    for sample in dataset:
        decord_vr, waveform_and_sr, video_id = sample
        combined_results, video_results, audio_results = model.predict(labels, decord_vr, waveform_and_sr, video_id)
        
        combined_candidates.append(combined_results)
        video_candidates.append(video_results)
        audio_candidates.append(audio_results)
    
    predictions = {"combined": combined_candidates,
                    "video": video_candidates,
                    "audio": video_candidates}
    
    metrices = calculate_metrices(video_dir_path, predictions, labels)

    wandb.log({
        "F_seg_a": metrices['F_seg_a'] ,
        "F_seg_v": metrices['F_seg_v'] ,
        "F_seg_av": metrices['F_seg_av'],
        "avg_type": metrices['avg_type'],
        "avg_event": metrices['avg_event'],
        "F_event_a": metrices['F_event_a'] ,
        "F_event_v": metrices['F_event_v'] ,
        "F_event_av": metrices['F_event_av'],
        "avg_type": metrices['avg_type'],
        "avg_event_level": metrices['avg_event_level'],
        "alpha": config.alpha,
        "filter_threshold": config.filter_threshold,
        "threshold_stage1": config.threshold_stage1,
        "threshold_stage2": config.threshold_stage2
    })

if __name__ == "__main__":
    hyperparameter_defaults = dict(
        sample_audio_sec = 2,
        alpha = 0.5,
        filter_threshold = 0.6,
        threshold_stage1 = 0.6,
        threshold_stage2 = 0.6
    )

    video_dir_path = ""
    audio_dir_path = ""

    wandb.init(config=hyperparameter_defaults, project="Video-Parsing-HPT")
    config = wandb.config
    sweep()