import json
import pandas as pd

def load_candidates(file_path):
    with open(file_path, 'r') as f:
        candidates = json.load(f)
    return candidates

def load_ground_truth(file_path):
    gts = pd.read_csv(file_path, delimiter='\t')
    return gts

def calculate_accuracy(candidates, gts):
    correct_count = 0
    total_count = 0

    for video_id, events in candidates.items():
        # Filter the ground truth data for the current video_id
        video_gts = gts[gts['filename'] == video_id]

        # Check each event in the candidate file
        for event, segments in events.items():
            # Filter ground truth segments for the current event
            gt_segments = video_gts[video_gts['event_labels'] == event]

            for candidate_segment in segments:
                # Check if the candidate segment is in the ground truth
                if any((gt_segments['onset'] == candidate_segment[0]) & (gt_segments['offset'] == candidate_segment[1])):
                    correct_count += 1

            total_count += len(segments)

    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy


def calc_metric(gts_file, candidates_file):
    candidates = load_candidates(candidates_file)
    gts = load_ground_truth(gts_file)

    # Calculate accuracy
    accuracy = calculate_accuracy(candidates, gts)
    print(f'Accuracy: {accuracy * 100:.2f}%')

calc_metric("/home/shaulov/work/zeroshot_AVE/intersection_results.csv", "cand.json")
