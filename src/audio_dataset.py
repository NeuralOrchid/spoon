# Dependencies
from typing import Optional, Literal
from pathlib import Path
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms as T


class BirdAudioDataset(Dataset):
    def __init__(
        self,
        audio_dir: Path,
        ann_file: str | Path,
        sample_rate: int,
        audio_length: int,
        dst: Optional[Literal['Bird-MY10', 'Bird-SEA10']] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.num_samples = audio_length * sample_rate
        self.audio_dir = audio_dir
        self.ann = pd.read_csv(ann_file)
        self.ann["label"] = self.ann["label"].astype("category").cat.codes

        if dst is not None:
            self.ann = self.ann[self.ann["dataset"] == dst]
            self.ann = self.ann.reset_index(drop=True)

        self.ann = self.ann.drop(columns="dataset")

        # ---------------------------------------------------------

        # Feature extraction
        self.mel = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=32,
            f_max=16_000,
            norm="slaney",
        ).to(self.device)

        self.db = T.AmplitudeToDB(
            stype="power",
            top_db=80,
        ).to(self.device)

        # SpecAugment
        self.freq_mask = T.FrequencyMasking(
            freq_mask_param=16
        ).to(self.device)

        self.time_mask = T.TimeMasking(
            time_mask_param=40
        ).to(self.device)


    def __len__(self) -> int:
        return len(self.ann)


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

    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            resampler = T.Resample(
                sr, self.sample_rate
            )
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = nn.functional.pad(signal, last_dim_padding)
        return signal
        
    def transform(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        if train:
            x = self._signal_augment(x)
        x = self.mel(x)
        if train:
            x = self._spec_augment(x)
        x = self.db(x)

        return x

    def __getitem__(self, idx: int):
        audio_path = self.audio_dir / self.ann.loc[idx, "file"]
        label = int(self.ann.loc[idx, "label"])

        signal, sr = torchaudio.load(audio_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        return signal, label
    

if __name__ == "__main__":
    # torch.Size([1, 128, 376])
    mydataset = BirdAudioDataset(
        audio_dir=Path('dataset/audio'),
        ann_file=Path('dataset/audio-annotations.csv'),
        sample_rate=32_000,
        audio_length=6,
    )

    img, _ = mydataset[0]
    img = mydataset.transform(img)
    print(img.shape)

    import matplotlib.pyplot as plt

    img = img.permute(1, 2, 0)

    plt.imshow(img)
    plt.axis("off")
    plt.show()