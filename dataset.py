from torch.utils.data import Dataset
import decord
import os
decord.bridge.set_bridge('torch')

class AVE(Dataset):
    def __init__(self, video_dir_path, annotations_file_path) -> None:
        self.video_dir_path = video_dir_path
        self.videos_ids = [video_id for video_id in os.listdir(video_dir_path) 
                           if os.path.splitext(os.path.join(self.video_dir_path, video_id))[1] == '.mp4']
        
        self.video_annotation_dict = {}
        with open(annotations_file_path, "r") as f:
            data = f.read().strip().split("\n")
        
        self.class2idx = {}
        for annotation in data[1:]:
            # Category&VideoID&Quality&StartTime&EndTime - example line in the annotations file
            category, video_id, _, start, end = annotation.split("&") 
            if category not in self.class2idx:
                self.class2idx[category] = len(self.class2idx)

            self.video_annotation_dict[f"{video_id}.mp4"] = {'class_name': category,
                                                    'class_idx': self.class2idx[category],
                                                    'start': int(start),
                                                    'end': int(end)}

    def __len__(self):
        return len(self.videos_ids)

    def __getitem__(self, idx):
        video_id = self.videos_ids[idx]
        vr = decord.VideoReader(os.path.join(self.video_dir_path, video_id))

        return vr[:], self.video_annotation_dict[video_id]


# if __name__ == '__main__':
#     dataset = AVE("/cortex/data/images/AVE_Dataset/AVE", 
#                   "/cortex/data/images/AVE_Dataset/Annotations.txt")
#     print("SDads")s