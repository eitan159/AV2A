import numpy as np

def iou(candidate, gt):
    intersection = max(0, min(candidate[1], gt[1]) - max(candidate[0], gt[0]))
    union = max(candidate[1], gt[1]) - min(candidate[0], gt[0])
    return intersection / union if union != 0 else 0

def compute_ap(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    return precision

def compute_map(candidates, gts, thresholds):
    ap_values = []
    for threshold in thresholds:
        tp, fp, fn = 0, 0, 0
        matched_candidates = set()
        matched_gts = set()
        
        for key in candidates:
            if key in gts:
                candidate = candidates[key]
                gt = gts[key]
                matched_candidates.add(key)
                matched_gts.add(key)
                if iou(candidate, gt) >= threshold:
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1
        
        for key in gts:
            if key not in matched_gts:
                fn += 1
        
        ap = compute_ap(tp, fp, fn)
        ap_values.append(ap)
    return np.mean(ap_values)




# candidates = {
#     "bark": [0, 9],
#     "cherch bell": [2, 8],
#     "sleep": [1, 10]
# }

# gts = {
#     "bark": [0, 10],
#     "cherch bell": [2, 10],
#     "ariel": [3,5]
# }

# thresholds_50_90 = np.arange(0.5, 1.0, 0.1)
# thresholds_10_90 = np.arange(0.1, 1.0, 0.1)

# mAP_50_90 = compute_map(candidates, gts, thresholds_50_90)
# average_mAP_10_90 = compute_map(candidates, gts, thresholds_10_90)

# print(f"mAP at [0.5:0.1:0.9]: {mAP_50_90}")
# print(f"Average mAP at [0.1:0.1:0.9]: {average_mAP_10_90}")
