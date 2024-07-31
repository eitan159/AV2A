import numpy as np
import json
import argparse

def load_gts(filename):
    gts = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            category, video_id, quality, start_time, end_time = line.strip().split('&')
            if video_id not in gts:
                gts[video_id] = []
            gts[video_id].append((category, int(start_time), int(end_time)))
    return gts

def iou(candidate, gt):
    intersection = max(0, min(candidate[1], gt[2]) - max(candidate[0], gt[1]))
    union = max(candidate[1], gt[2]) - min(candidate[0], gt[1])
    return intersection / union if union != 0 else 0

def compute_ap(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    return precision

def compute_map(candidates, gts, thresholds):
    ap_values = []
    for threshold in thresholds:
        tp, fp, fn = 0, 0, 0
        for video_id in candidates:
            if video_id in gts:
                for category, candidate_times in candidates[video_id].items():
                    for candidate_time in candidate_times:
                        matched = False
                        for gt in gts[video_id]:
                            if gt[0] == category and iou(candidate_time, gt) >= threshold:
                                tp += 1
                                matched = True
                                break
                        if not matched:
                            fp += 1
            else:
                fp += len(candidates[video_id])
        
        for video_id in gts:
            if video_id not in candidates:
                fn += len(gts[video_id])
        
        ap = compute_ap(tp, fp, fn)
        ap_values.append(ap)
    return ap_values


def calc_accuracy(candidates, gts):
    counter = 0
    hits = 0
    for video_id in candidates:
        for category, candidate_times in candidates[video_id].items():
            for candidate_time in candidate_times:
                for gt in gts[video_id]:
                    if gt[0] == category and iou(candidate_time, gt) == 1:
                        hits += 1 
                        break
            counter += max(len(gts[video_id]), len(candidate_times))
                    
    return hits / counter

def calc_metrics(annotation_file_path, candidates_file_path):
    with open(candidates_file_path, 'r') as f:
        candidates = json.load(f)
        
    gts = load_gts(annotation_file_path)

    # Filter candidates and gts to only include matching video IDs
    filtered_candidates = {k: v for k, v in candidates.items() if k in gts}
    filtered_gts = {k: v for k, v in gts.items() if k in candidates}

    print(f"Accuracy: {calc_accuracy(filtered_candidates, filtered_gts)}")

    # thresholds_50_100 = np.linspace(0.5, 1.0, 6)
    thresholds_50_90 = np.arange(0.5, 1.0, 0.1)
    thresholds_10_90 = np.arange(0.1, 1.0, 0.1)

    ap_values_50_90 = compute_map(filtered_candidates, filtered_gts, thresholds_50_90)
    average_mAP_10_90 = np.mean(compute_map(filtered_candidates, filtered_gts, thresholds_10_90))

    for threshold, ap_value in zip(thresholds_50_90, ap_values_50_90):
        print(f"AP at {threshold}: {ap_value}")

    print(f"Average mAP at [0.1:0.1:0.9]: {average_mAP_10_90}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_file_path', required=True, type=str)
    parser.add_argument('--candidates_file_path', required=True, type=str)
    args = parser.parse_args()

    # calc_metrics(args.annotations_file_path, args.candidates_file_path)
    calc_metrics("/media/data2/shaulov/AVE_Dataset/testSet.txt", "cand.json")