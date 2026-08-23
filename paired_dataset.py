# Dependencies
import random
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset



class AudioTransform:
    def __init__(
            self,
            sample_rate:int=32_000,
            audio_length:float=6.128,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.num_samples = int(audio_length * sample_rate)

        # Feature extraction
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=32,
            f_max=16_000,
            norm="slaney",
        ).to(self.device)

        self.db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80,
        ).to(self.device)

        # SpecAugment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=16
        ).to(self.device)

        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=40
        ).to(self.device)

    def _random_time_shift(self, signal):
        if random.random() >= 0.5:
            return signal
        length = signal.shape[-1]
        shift = random.randint(0, length - 1)
        return torch.roll(signal, shifts=shift, dims=-1)

    def _random_gain(self, signal):
        if random.random() >= 0.3:
            return signal
        gain_db = random.uniform(-6.0, 6.0)
        gain = 10 ** (gain_db / 20.0)
        return signal * gain

    def _random_noise(self, signal):
        if random.random() >= 0.4:
            return signal
        noise = torch.randn_like(signal)
        signal_power = signal.pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_db = random.uniform(10.0, 30.0)
        snr_linear = 10 ** (snr_db / 10.0)
        target_noise_power = signal_power / snr_linear
        noise = noise * torch.sqrt(
            target_noise_power / (noise_power + 1e-8)
        )

        return signal + noise

    # Spectrogram augmentation
    def _signal_augment(self, signal):
        signal = self._random_time_shift(signal)
        signal = self._random_gain(signal)
        signal = self._random_noise(signal)
        return signal

    def _spec_augment(self, mel):
        if random.random() < 0.2:
            mel = self.freq_mask(mel)
        if random.random() < 0.2:
            mel = self.time_mask(mel)
        return mel
        
    def __call__(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        if train:
            x = self._signal_augment(x)
        x = self.mel(x)
        if train:
            x = self._spec_augment(x)
        x = self.db(x)

        return x



class PairedDataset(Dataset):
    def __init__(
        self,
        dir: Path = Path('dataset'),
        ann_file: str | Path = Path('dataset/paired-annotations.csv'),
        dst: Optional[Literal[
            'Bird-MY10',    # Train
            'Bird-SEA10',   # Val
        ]] = None,
    ):
        # Dataset Folders/Directories 
        self.image_dir = dir / 'image'
        self.audio_dir = dir / 'audio'

        # Annotations file
        self.ann = pd.read_csv(ann_file)
        self.ann["label"] = self.ann["label"].astype("category").cat.codes

        if dst is not None:
            self.ann = self.ann[self.ann["dataset"] == dst]
            self.ann = self.ann.reset_index(drop=True)

        self.ann = self.ann.drop(columns="dataset")

        if dst == 'Bird-SEA10':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(384),
                torchvision.transforms.RandomResizedCrop(256, (0.8, 1.0), (0.95, 1.05)),
                torchvision.transforms.RandomHorizontalFlip(),
    
                torchvision.transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1
                ),
    
                torchvision.transforms.ToTensor(),
    
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ])


    def _pad2square(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        m = max(w, h)
        pl = (m - w) // 2
        pt = (m - h) // 2
        pr = m - w - pl
        pb = m - h - pt
        return ImageOps.expand(img, border=(pl, pt, pr, pb))

    def _transform_if_necessary(self, signal:torch.Tensor, sr:int) -> torch.Tensor:
        # Resample if necessary
        if sr != 32_000:
            resampler = torchaudio.transforms.Resample(sr, 32_000)
            signal = resampler(signal)
        # Mix down if necessary
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        # Cut if necessary
        if signal.shape[1] > 196096:
            signal = signal[:, :196096]
        length_signal = signal.shape[1]
        # Right pad if necessary
        if length_signal < 196096:
            num_missing_samples = 196096 - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = nn.functional.pad(signal, last_dim_padding)
        return signal


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_dir / self.ann.loc[idx, "image_file"]
        audio_path = self.audio_dir / self.ann.loc[idx, "audio_file"]
        label = int(self.ann.loc[idx, "label"])

        image = Image.open(image_path).convert("RGB")
        image = self._pad2square(image)
        image = self.transform(image)

        signal, sr = torchaudio.load(audio_path)
        audio = self._transform_if_necessary(signal, sr)

        # image: [3, 256, 256]
        # audio: [1, 128, 384]
        # label: 0

        return image, audio, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_dataset   = PairedDataset(dst='Bird-MY10')
    val_dataset     = PairedDataset(dst='Bird-SEA10')

    image, audio, label = val_dataset[0]
    transform = AudioTransform()
    audio = transform(audio, False)
    
    print("image:", image.shape)
    print("audio:", audio.shape)
    print("label:", label)

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