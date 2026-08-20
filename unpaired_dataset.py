# Dependencies
from pathlib import Path
from collections import defaultdict
import random

import pandas as pd
from torch.utils.data import Dataset

try:
    from . import (
        BirdImageDataset,
        BirdAudioDataset,
    )
except:
    from audio_dataset import BirdAudioDataset
    from image_dataset import BirdImageDataset



class ClassIndex:
    def __init__(self, dataset: BirdImageDataset | BirdAudioDataset):
        self.by_class = defaultdict(list)

        for idx, label in enumerate(dataset.ann["label"]):
            self.by_class[int(label)].append(idx)

        self.classes = sorted(self.by_class.keys())

    def sample(self, label):
        return random.choice(self.by_class[label])



class UnpairedDataset(Dataset):
    def __init__(
        self,
        image_dataset: BirdImageDataset,
        audio_dataset: BirdAudioDataset,
        samples_per_epoch: int = None,
    ):
        self.image_dataset = image_dataset
        self.audio_dataset = audio_dataset

        self.image_index = ClassIndex(image_dataset)
        self.audio_index = ClassIndex(audio_dataset)

        self.num_classes = sorted(
            set(self.image_index.classes)
            & set(self.audio_index.classes)
        ).__len__()

        if samples_per_epoch is None:
            self.samples_per_epoch = max(
                len(image_dataset),
                len(audio_dataset),
            )

        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int):

        # Choose a class first
        label = idx % self.num_classes

        # Independently choose image/audio from that class (unpaired)
        image_idx = self.image_index.sample(label)
        audio_idx = self.audio_index.sample(label)

        image, image_label = self.image_dataset[image_idx]
        audio, audio_label = self.audio_dataset[audio_idx]

        assert image_label == audio_label == label

        return image, audio, label



if __name__ == "__main__":
    imagedataset = BirdImageDataset(
        img_dir=Path('dataset/image'),
        ann_file=Path('dataset/image-annotations.csv'),
        train=False
    )

    audiodataset = BirdAudioDataset(
        audio_dir=Path('dataset/audio'),
        ann_file=Path('dataset/audio-annotations.csv'),
        sample_rate=32_000,
        audio_length=6.128,
    )

    unpaireddataset = UnpairedDataset(
        image_dataset=imagedataset,
        audio_dataset=audiodataset,
    )

    image, audio, label = unpaireddataset[0]
    audio = audiodataset.transform(audio)
    
    print("image:", image.shape)
    print("audio:", audio.shape)
    print("label:", label)

    import torch
    import matplotlib.pyplot as plt

    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    image = image * std[:, None, None] + mean[:, None, None]
    image = image.permute(1, 2, 0)
    image = image.clamp(0, 1)

    audio = audio.permute(1, 2, 0)


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Bird Image")

    axes[1].imshow(audio)
    axes[1].axis("off")
    axes[1].set_title("Log-Mel Spectrogram")

    plt.tight_layout()
    plt.show()