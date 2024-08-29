import json
from dataset import LLP
from tqdm import tqdm
import argparse
from eval_metrics import calculate_metrices_LLP, calculate_metrices_AVE, print_metrices, calculate_ave_acc
from models.languagebindmodel.languagebind import LanguageBind, LanguageBindImageTokenizer, to_device
from video_parser_optmizer import VideoParserOptimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', required=True, type=str)    
    parser.add_argument('--audio_dir_path', required=True, type=str)
    parser.add_argument('--gpu_id', default=-1, type=int)
    parser.add_argument('--threshold_stage1', default=0.6, type=float)
    parser.add_argument('--threshold_stage2', default=0.6, type=float)
    parser.add_argument('--filter_threshold', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.8, type=float)
    parser.add_argument('--candidates_file_path', required=True, type=str)
    parser.add_argument('--sample_audio_sec', default=2, type=int)
    parser.add_argument('--dataset', default='LLP', type=str, choices=['LLP', 'AVE'])
    parser.add_argument('--method', default='BBSE', type=str)
    parser.add_argument('--without_filter_classes', action="store_true")
    parser.add_argument('--without_refine_segments', action="store_true")
    args = parser.parse_args()

    threshold_stage1 = args.threshold_stage1
    threshold_stage2 = args.threshold_stage2

    filter_threshold = args.filter_threshold
    alpha = args.alpha
    gamma = args.gamma
    video_dir_path = args.video_dir_path
    
    if args.dataset == "LLP":
        labels = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                    'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                    'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                    'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                    'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                    'Clapping']
        subset = None

    elif args.dataset == "AVE":
        with open("./test_AVE.json", 'r') as f:
            subset = json.load(f)

        labels = []
        for k, v in subset.items():
            labels.extend([sample['class'] for sample in v])
        
        labels = list(set(labels))

    dataset = LLP(args.video_dir_path,
                  args.audio_dir_path,
                  subset=subset)

    device = f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cpu"

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

    model = VideoParserOptimizer(args.method, model, tokenizer, labels, device, args.sample_audio_sec, alpha, 
                            filter_threshold, threshold_stage1, threshold_stage2, gamma, args.without_filter_classes,
                            args.without_refine_segments, args.dataset)

    combined_candidates, video_candidates, audio_candidates = [], [], []
    for sample in tqdm(dataset, desc="Processing samples"):
        decord_vr, waveform_and_sr, video_id = sample
        combined_results, video_results, audio_results = model.predict(labels, decord_vr, waveform_and_sr, video_id)
        
        combined_candidates.append(combined_results)
        video_candidates.append(video_results)
        audio_candidates.append(audio_results)
    
    predictions = {"combined": combined_candidates,
                    "video": video_candidates,
                    "audio": video_candidates}
    
    with open(args.candidates_file_path, 'w') as f:
        json.dump(predictions, f)

    if args.dataset == "LLP":
        print_metrices(calculate_metrices_LLP(args.video_dir_path, predictions, labels))
    elif args.dataset == "AVE":
        print(calculate_ave_acc(predictions, labels, subset))
        print_metrices(calculate_metrices_AVE(predictions, labels, subset))

