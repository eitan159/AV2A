import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import json
from eval_metrics import calculate_metrices_LLP

def create_hist_v2(data1, data2, categories, metric_name, label1="with", label2="without"):
    # Plotting the bar chart
    plt.figure(figsize=(15, 15))

    width = 0.35
    # Plotting
    fig, ax = plt.subplots()

    # Plot precision bars
    bar1 = ax.barh(np.arange(len(categories)) - width/2, data1, width, label=label1, edgecolor='black')

    # Plot recall bars
    bar2 = ax.barh(np.arange(len(categories)) + width/2, data2, width, label=label2, edgecolor='black')

    # Adding labels and title
    ax.set_ylabel('labels')
    ax.set_xlabel(f'{metric_name}')
    ax.set_title(f'{metric_name} bar plot')

    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, ha='right', fontsize=8)
    plt.subplots_adjust(bottom=0.3)
    # ax.set_xticks(categories, [c for c in categories], rotation=45, ha='right')  # Rotate x-axis labels
    ax.legend(fontsize="small")

    # Display the plot
    plt.tight_layout()
    plt.savefig(f'{metric_name}_bar_plot.png')


def create_hist(data1, data2, categories, metric_name, label1="with", label2="without"):
    # Plotting the bar chart
    plt.figure(figsize=(12, 12))

    width = 0.35
    # Plotting
    fig, ax = plt.subplots()

    # Plot precision bars
    bar1 = ax.bar(np.arange(len(categories)) - width/2, data1, width, label=label1, edgecolor='black')

    # Plot recall bars
    bar2 = ax.bar(np.arange(len(categories)) + width/2, data2, width, label=label2, edgecolor='black')

    # Adding labels and title
    ax.set_xlabel('labels')
    ax.set_ylabel(f'{metric_name}')
    ax.set_title(f'{metric_name} bar plot')

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=90, ha='right')
    #plt.subplots_adjust(bottom=0.3)
    # ax.set_xticks(categories, [c for c in categories], rotation=45, ha='right')  # Rotate x-axis labels
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig(f'{metric_name}_bar_plot.png')

def create_hist_avg_sec(data, categories, metric_name):
    # Plotting the bar chart
    plt.figure(figsize=(12, 12))
    # Plotting
    fig, ax = plt.subplots()

    # Plot precision bars
    bar1 = ax.bar(np.arange(len(categories)), data, edgecolor='black')

    # Plot recall bars
    # Adding labels and title
    ax.set_xlabel('labels')
    ax.set_ylabel(f'Avg seconds')
    ax.set_title(f'{metric_name} bar plot')

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    #plt.subplots_adjust(bottom=0.3)
    # ax.set_xticks(categories, [c for c in categories], rotation=45, ha='right')  # Rotate x-axis labels
    

    # Display the plot
    plt.tight_layout()
    plt.savefig(f'{metric_name}_bar_plot.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', required=True, type=str)
    parser.add_argument('--dataset', default='LLP', type=str)
    args = parser.parse_args()

    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                        'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                        'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                        'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                        'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                        'Clapping']

    with open("/home/lab/eitanshaar/zeroshot_AVE/results/candidates.json", 'r') as f:
            pred = json.load(f)

    metrices_best, metric_per_class_best = calculate_metrices_LLP(args.video_dir_path, pred, categories)


    with open("/home/lab/eitanshaar/zeroshot_AVE/results/candidates-without-BBSE.json", 'r') as f:
        pred = json.load(f)

    metrices, metric_per_class = calculate_metrices_LLP(args.video_dir_path, pred, categories)


    create_hist_v2([metric_per_class_best[m]['F_seg_av'] for m in metric_per_class_best], [metric_per_class[m]['F_seg_av'] for m in metric_per_class], categories, "Audio-Visual-segment-level", label1="BBSE", label2="W/O BBSE")
    create_hist_v2([metric_per_class_best[m]['F_event_av'] for m in metric_per_class_best], [metric_per_class[m]['F_event_av'] for m in metric_per_class], categories, "Audio-Visual-event-level", label1="BBSE", label2="W/O BBSE")
    create_hist_avg_sec([metric_per_class_best[m]['avg_sec'] for m in metric_per_class_best], categories, "AVG-sec-per-event")