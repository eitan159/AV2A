from PIL import Image
from models.imagebindmodel.imagebind.data import load_and_transform_text, load_and_transform_audio_data
import torch
from models.imagebindmodel.imagebind.models import imagebind_model
from models.imagebindmodel.imagebind.models.imagebind_model import ModalityType
from dataset import AVE
from torchvision import transforms
from torch.utils.data import DataLoader


image_transforms_imagebind = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

def predict(labels, frames, audio_file):

    inputs = {
        ModalityType.TEXT: load_and_transform_text(labels, device),
        ModalityType.VISION: frames.to(device),
        # ModalityType.AUDIO: load_and_transform_audio_data([audio_file], device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    print(
        "Vision x Text: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )
    print(
        "Audio x Text: ",
        torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )
    print(
        "Vision x Audio: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
    )

if __name__ == '__main__':
    dataset = AVE("/cortex/data/images/AVE_Dataset/AVE", 
                  "/cortex/data/images/AVE_Dataset/Annotations.txt",
                  frames_transforms=image_transforms_imagebind)
    
    labels = list(dataset.class2idx.keys())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    for sample in dataset:
        frames, audio_file_name, label_dict = sample
        predict(labels, frames, audio_file_name)

    
