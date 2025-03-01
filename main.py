import json
from dataset import VideoDataset
from tqdm import tqdm
import argparse
from eval_metrics import calculate_metrices_LLP, calculate_metrices_AVE, print_metrices, calculate_ave_acc
from video_parser_optmizer import VideoParserOptimizer
from utils import set_random_seed, load_data
from time import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', required=True, type=str)    
    parser.add_argument('--audio_dir_path', required=True, type=str)
    parser.add_argument('--gpu_id', default=-1, type=int)
    parser.add_argument('--backbone', default='language_bind', type=str, choices=['language_bind', 'clip_clap'])
    parser.add_argument('--threshold_stage1', default=0.6, type=float)
    parser.add_argument('--threshold_stage2', default=0.6, type=float)
    parser.add_argument('--filter_threshold', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.8, type=float)
    parser.add_argument('--candidates_file_path', required=True, type=str)
    parser.add_argument('--sample_audio_sec', default=2, type=int)
    parser.add_argument('--dataset', default='LLP', type=str, choices=['LLP', 'AVE'])
    parser.add_argument('--method', default='bbse', type=str, choices=['bbse', 'bbse-cosine', 'cosine', 'naive'])
    parser.add_argument('--without_filter_classes', action="store_true")
    parser.add_argument('--without_refine_segments', action="store_true")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--labels_shift_iters', default=1, type=int)
    parser.add_argument('--fusion', default='early', type=str, choices=['early', 'late'])
    args = parser.parse_args()

    set_random_seed(args.seed)

    threshold_stage1 = args.threshold_stage1
    threshold_stage2 = args.threshold_stage2

    filter_threshold = args.filter_threshold
    alpha = args.alpha
    gamma = args.gamma
    video_dir_path = args.video_dir_path
    
    device = f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cpu"

    subset, labels = load_data(args.dataset)

    dataset = VideoDataset(args.video_dir_path,
                  args.audio_dir_path,
                  args.backbone,
                  subset=subset)

    video_parser_optimizer = VideoParserOptimizer(args.method, args.backbone, labels, device, alpha, 
                        filter_threshold, threshold_stage1, threshold_stage2, gamma, args.without_filter_classes,
                        args.without_refine_segments, args.dataset, args.fusion)


    combined_candidates, video_candidates, audio_candidates = [], [], []
    total_time = 0
    for sample in tqdm(dataset, desc="Processing samples"):
        decord_vr, waveform_and_sr, video_id = sample
        
        start = time()
        combined_results, video_results, audio_results = video_parser_optimizer(labels, decord_vr, waveform_and_sr, video_id)
        
        total_time += time() - start 

        combined_candidates.append(combined_results)
        video_candidates.append(video_results)
        audio_candidates.append(audio_results)
    
    predictions = {"combined": combined_candidates,
                    "video": video_candidates,
                    "audio": video_candidates}
    
    with open(args.candidates_file_path, 'w') as f:
        json.dump(predictions, f)

    if args.dataset == "LLP":
        metrices, _ = calculate_metrices_LLP(args.video_dir_path, predictions, labels)
        print_metrices(metrices)
    elif args.dataset == "AVE":
        print(calculate_ave_acc(predictions, labels, subset))
    
    print(f"Avg. predict time: {total_time / len(dataset)}")

