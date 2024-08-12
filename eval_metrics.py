import numpy as np
import argparse
import json
import pandas as pd
import os

def Precision(X_pre, X_gt):

    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x*y)/np.sum(x)
    return p/N


def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(y)
    return p/N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2*np.sum(x * y) / (np.sum(x) + np.sum(y))
    return p/N

def event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # extract events
    N = 25
    event_p_a = [None for n in range(25)]
    event_gt_a = [None for n in range(25)]
    event_p_v = [None for n in range(25)]
    event_gt_v = [None for n in range(25)]
    event_p_av = [None for n in range(25)]
    event_gt_av = [None for n in range(25)]

    TP_a = np.zeros(25)
    TP_v = np.zeros(25)
    TP_av = np.zeros(25)

    FP_a = np.zeros(25)
    FP_v = np.zeros(25)
    FP_av = np.zeros(25)

    FN_a = np.zeros(25)
    FN_v = np.zeros(25)
    FN_av = np.zeros(25)

    for n in range(N):
        seq_pred = SO_a[n, :]
        if np.sum(seq_pred)!=0:
            x = extract_event(seq_pred, n)
            event_p_a[n] = x
        seq_gt = GT_a[n, :]
        if np.sum(seq_gt)!=0:
            x = extract_event(seq_gt, n)
            event_gt_a[n] = x

        seq_pred = SO_v[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_v[n] = x
        seq_gt = GT_v[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_v[n] = x

        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_av[n] = x

        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_av[n] = x

        tp, fp, fn = event_wise_metric(event_p_a[n], event_gt_a[n])
        TP_a[n] += tp
        FP_a[n] += fp
        FN_a[n] += fn

        tp, fp, fn = event_wise_metric(event_p_v[n], event_gt_v[n])
        TP_v[n] += tp
        FP_v[n] += fp
        FN_v[n] += fn

        tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n])
        TP_av[n] += tp
        FP_av[n] += fp
        FN_av[n] += fn

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP_a)
    F_a = []
    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a.append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_a) == 0:
        f_a = 1.0 # all true negatives
    else:
        f_a = (sum(F_a)/len(F_a))

    if len(F_v) == 0:
        f_v = 1.0 # all true negatives
    else:
        f_v = (sum(F_v)/len(F_v))

    if len(F) == 0:
        f = 1.0 # all true negatives
    else:
        f = (sum(F)/len(F))
    if len(F_av) == 0:
        f_av = 1.0 # all true negatives
    else:
        f_av = (sum(F_av)/len(F_av))

    return f_a, f_v, f, f_av


def segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # compute F scores
    TP_a = np.sum(SO_a * GT_a, axis=1)
    FN_a = np.sum((1-SO_a)*GT_a, axis = 1)
    FP_a = np.sum(SO_a*(1-GT_a),axis=1)

    n = len(FP_a)
    F_a = []
    for ii in range(n):
        if (TP_a+FP_a)[ii]!= 0 or (TP_a+FN_a)[ii]!= 0:
            F_a.append(2*TP_a[ii] / (2*TP_a[ii] + (FN_a + FP_a)[ii]))

    TP_v = np.sum(SO_v * GT_v, axis=1)
    FN_v = np.sum((1 - SO_v) * GT_v, axis=1)
    FP_v = np.sum(SO_v * (1 - GT_v), axis=1)
    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP)

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    n = len(FP_av)
    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))


    if len(F_a) == 0:
        f_a = 1.0 # all true negatives
    else:
        f_a = (sum(F_a)/len(F_a))

    if len(F_v) == 0:
        f_v = 1.0 # all true negatives
    else:
        f_v = (sum(F_v)/len(F_v))

    if len(F) == 0:
        f = 1.0 # all true negatives
    else:
        f = (sum(F)/len(F))
    if len(F_av) == 0:
        f_av = 1.0 # all true negatives
    else:
        f_av = (sum(F_av)/len(F_av))

    return f_a, f_v, f, f_av


def to_vec(start, end):
    x = np.zeros(10)
    for i in range(start, end):
        x[i] = 1
    return x

def extract_event(seq, n):
    x = []
    i = 0
    while i < 10:
        if seq[i] == 1:
            start = i
            if i + 1 == 10:
                i = i + 1
                end = i
                x.append(to_vec(start, end))
                break

            for j in range(i + 1, 10):
                if seq[j] != 1:
                    i = j + 1
                    end = j
                    x.append(to_vec(start, end))
                    break
                else:
                    i = j + 1
                    if i == 10:
                        end = i
                        x.append(to_vec(start, end))
                        break
        else:
            i += 1
    return x

def event_wise_metric(event_p, event_gt):
    TP = 0
    FP = 0
    FN = 0

    if event_p is not None:
        num_event = len(event_p)
        for i in range(num_event):
            x1 = event_p[i]
            if event_gt is not None:
                nn = len(event_gt)
                flag = True
                for j in range(nn):
                    x2 = event_gt[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2): #0.5
                        TP += 1
                        flag = False
                        break
                if flag:
                    FP += 1
            else:
                FP += 1

    if event_gt is not None:
        num_event = len(event_gt)
        for i in range(num_event):
            x1 = event_gt[i]
            if event_p is not None:
                nn = len(event_p)
                flag = True
                for j in range(nn):
                    x2 = event_p[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2): #0.5
                        flag = False
                        break
                if flag:
                    FN += 1
            else:
                FN += 1
    return TP, FP, FN

def calculate_metrices(video_dir_path, pred, categories, split="test"):

    id_to_idx = {id: index for index, id in enumerate(categories)}
    
    download_videos_ids = [video_id.replace(".mp4", "") for video_id in os.listdir(video_dir_path) 
                           if os.path.splitext(os.path.join(video_dir_path, video_id))[1] == '.mp4']

    
    pred_combined = {list(d.keys())[0]: list(d.values())[0] for d in pred["combined"]}
    pred_video = {list(d.keys())[0]: list(d.values())[0] for d in pred["video"]}
    pred_audio = {list(d.keys())[0]: list(d.values())[0] for d in pred["audio"]}


    df = pd.read_csv(f"annotations/AVVP_{split}_pd.csv", header=0, sep='\t')
    df_a = pd.read_csv("annotations/AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv("annotations/AVVP_eval_visual.csv", header=0, sep='\t')
    
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []
    for file_name in df["filename"].values:
        
        parts = file_name.split('_')
        # Join all parts except the last two
        new_file_name = '_'.join(parts[:-2])

        if new_file_name not in download_videos_ids:
            continue

        SO_a = np.zeros((25, 10))
        SO_v = np.zeros((25, 10))
        SO_av = np.zeros((25, 10))
        GT_a = np.zeros((25, 10))
        GT_v = np.zeros((25, 10))
        GT_av = np.zeros((25, 10))

        df_vid_a = df_a.loc[df_a['filename'] == file_name]
        filenames = df_vid_a["filename"]
        events = df_vid_a["event_labels"]
        onsets = df_vid_a["onset"]
        offsets = df_vid_a["offset"]
        num = len(filenames)
        if num > 0:
            for i in range(num):
                x1 = int(onsets[df_vid_a.index[i]])
                x2 = int(offsets[df_vid_a.index[i]])
                event = events[df_vid_a.index[i]]
                idx = id_to_idx[event]
                GT_a[idx, x1:x2] = 1
        

        df_vid_v = df_v.loc[df_v['filename'] == file_name]
        filenames = df_vid_v["filename"]
        events = df_vid_v["event_labels"]
        onsets = df_vid_v["onset"]
        offsets = df_vid_v["offset"]
        num = len(filenames)
        if num > 0:
            for i in range(num):
                x1 = int(onsets[df_vid_v.index[i]])
                x2 = int(offsets[df_vid_v.index[i]])
                event = events[df_vid_v.index[i]]
                idx = id_to_idx[event]
                GT_v[idx, x1:x2] = 1
        
        GT_av = GT_a * GT_v

        if new_file_name in pred_combined:
            for pred_dict in pred_combined[new_file_name]:
                idx, x1, x2 = id_to_idx[pred_dict["event_label"].capitalize()], pred_dict["start"], pred_dict["end"]
                SO_av[idx, x1:x2] = 1

        if new_file_name in pred_video:
            for pred_dict in pred_video[new_file_name]:
                idx, x1, x2 = id_to_idx[pred_dict["event_label"].capitalize()], pred_dict["start"], pred_dict["end"]
                SO_v[idx, x1:x2] = 1
        
        if new_file_name in pred_audio:
            for pred_dict in pred_audio[new_file_name]:
                idx, x1, x2 = id_to_idx[pred_dict["event_label"].capitalize()], pred_dict["start"], pred_dict["end"]
                SO_a[idx, x1:x2] = 1

        # segment-level F1 scores
        f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
        F_seg_a.append(f_a)
        F_seg_v.append(f_v)
        F_seg.append(f)
        F_seg_av.append(f_av)

        # event-level F1 scores
        f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
        F_event_a.append(f_a)
        F_event_v.append(f_v)
        F_event.append(f)
        F_event_av.append(f_av)
    
    metrices = {}
    metrices['F_seg_a'] = 100 * np.mean(np.array(F_seg_a))
    metrices['F_seg_v'] = 100 * np.mean(np.array(F_seg_v))
    metrices['F_seg_av'] = 100 * np.mean(np.array(F_seg_av))
    metrices['avg_type'] = (100 * np.mean(np.array(F_seg_av)) + 100 * np.mean(np.array(F_seg_a)) + 100 * np.mean(np.array(F_seg_v))) / 3.
    metrices['avg_event'] = 100 * np.mean(np.array(F_seg))

    metrices['F_event_a'] = 100 * np.mean(np.array(F_event_a))
    metrices['F_event_v'] = 100 * np.mean(np.array(F_event_v))
    metrices['F_event_av'] = 100 * np.mean(np.array(F_event_av))
    metrices['avg_type_event'] = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(np.array(F_event_v))) / 3.
    metrices['avg_event_level'] = 100 * np.mean(np.array(F_event))

    return metrices

def print_metrices(metrices):

    print('Audio Event Detection Segment-level F1: {:.1f}'.format(metrices['F_seg_a']))
    print('Visual Event Detection Segment-level F1: {:.1f}'.format(metrices['F_seg_v']))
    print('Audio-Visual Event Detection Segment-level F1: {:.1f}'.format(metrices['F_seg_av']))

    print('Segment-levelType@Avg. F1: {:.1f}'.format(metrices['avg_type']))
    print('Segment-level Event@Avg. F1: {:.1f}'.format(metrices['avg_event']))

    print('Audio Event Detection Event-level F1: {:.1f}'.format(metrices['F_event_a']))
    print('Visual Event Detection Event-level F1: {:.1f}'.format(metrices['F_event_v']))
    print('Audio-Visual Event Detection Event-level F1: {:.1f}'.format(metrices['F_event_av']))

    print('Event-level Type@Avg. F1: {:.1f}'.format(metrices['avg_type_event']))
    print('Event-level Event@Avg. F1: {:.1f}'.format(metrices['avg_event_level']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', required=True, type=str)
    parser.add_argument('--predictions_json_file_path', required=True, type=str)
    args = parser.parse_args()

    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    
    with open(args.predictions_json_file_path, 'r') as f:
        pred = json.load(f)

    print_metrices(calculate_metrices(args.video_dir_path, pred, categories))
    
            


