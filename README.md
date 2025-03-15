# Adapting to the Unknown: Training-Free Audio-Visual Event Perception with Dynamic Thresholds [CVPR2025]
This repo is the official code of ["Adapting to the Unknown: Training-Free Audio-Visual Event Perception with Dynamic Thresholds"](https://arxiv.org/pdf/2404.03179.pdf). TODO replace to our arxiv link


## Abstract
In the domain of audio-visual event perception, which focuses on the temporal localization and classification of events across distinct modalities (audio and visual), existing approaches are constrained by the vocabulary available in their training data. This limitation significantly impedes their capacity to generalize to novel, unseen event categories. Furthermore, the annotation process for this task is labor-intensive, requiring extensive manual labeling across modalities and temporal segments, limiting the scalability of current methods. Current state-of-the-art models ignore the shifts in event distributions over time, reducing their ability to adjust to changing video dynamics. Additionally, previous methods rely on late fusion to combine audio and visual information. While straightforward, this approach results in a significant loss of multimodal interactions.  
To address these challenges, we propose **A**udio-**V**isual **A**daptive **V**ideo **A**nalysis (AV²A), a model-agnostic approach that requires no further training and integrates an score-level fusion technique to retain richer multimodal interactions. AV²A also includes a within-video label shift algorithm, leveraging input video data and predictions from prior frames to dynamically adjust event distributions for subsequent frames. Moreover, we present the first training-free, open-vocabulary baseline for audio-visual event perception, demonstrating that AV²A achieves substantial improvements over naive training-free baselines. We demonstrate the effectiveness of AV²A on both zero-shot and weakly-supervised state-of-the-art methods, achieving notable improvements in performance metrics over existing approaches. 

## Requirements
````
conda env create -f environment.yml
````

## Run
````
# LanguageBind
python main.py --video_dir_path "" --audio_dir_path "" --gpu_id 0 --backbone language_bind --candidates_file_path "" --alpha 0.5 --filter_threshold 0.55 --threshold_stage1 0.75 --threshold_stage2 0.75 --gamma 2.5 --dataset LLP/AVE --method naive --fusion early

# CLIP & CLAP
python main.py --video_dir_path "" --audio_dir_path "" --gpu_id 0 --backbone clip_clap --candidates_file_path "" --alpha 0.45 --filter_threshold 0.5 --threshold_stage1 0.75 --threshold_stage2 0.75 --gamma 1 --dataset LLP/AVE --method naive --fusion early
````

## Citation
If you find our code useful for your research, please consider citing our paper.
```
Our CITE
```