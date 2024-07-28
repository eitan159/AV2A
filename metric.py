import numpy as np

def iou(candidate, gt):
    intersection = max(0, min(candidate[1], gt[1]) - max(candidate[0], gt[0]))
    union = max(candidate[1], gt[1]) - min(candidate[0], gt[0])
    return intersection / union if union != 0 else 0

def compute_ap(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) != 0 else 0 
    return precision

def compute_map(candidates, gts, thresholds):
    ap_values = []
    for threshold in thresholds:
        tp, fp, fn = 0, 0, 0
        for key in candidates:
            candidate = candidates[key]
            gt = gts[key]
            if iou(candidate, gt) >= threshold:
                tp += 1
            else:
                fp += 1
            fn += 1
        ap = compute_ap(tp, fp, fn)
        ap_values.append(ap)
    return np.mean(ap_values)



###########NEED TO CHENGE THE INPUTS ONES WE'LL UNDERSTAND HOW WE ARE GOING TO COMPARE THE CANDIDATES WITH THE GT'S, THIS IS ONLY A POC IMPLEMENTATION##############
candidates = {
    "bark": [0, 9],
    "cherch bell": [2, 8]
}

gts = {
    "bark": [0, 10],
    "cherch bell": [2, 10]
}

thresholds_50_90 = np.arange(0.5, 1.0, 0.1)
thresholds_10_90 = np.arange(0.1, 1.0, 0.1)

mAP_50_90 = compute_map(candidates, gts, thresholds_50_90)
average_mAP_10_90 = compute_map(candidates, gts, thresholds_10_90)

print(f"mAP at [0.5:0.1:0.9]: {mAP_50_90}")
print(f"Average mAP at [0.1:0.1:0.9]: {average_mAP_10_90}")
