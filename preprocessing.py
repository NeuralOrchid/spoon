# Dependencies
from pathlib import Path
import re
import csv
import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from tqdm import tqdm


# Preprocessing audio
class AudioPreprocessing:
    def __init__(self, root: Path = Path("dataset"), window_sec: int = 8):
        
        self.window_sec:int = window_sec
        self.count:int = 0
        self.root:Path = root 


    def _normalize(self, x: np.ndarray) -> np.ndarray:
        low = np.percentile(x, 10)
        high = np.percentile(x, 90)
    
        if high - low < 1e-10:
            return np.zeros_like(x)
    
        x = (x - low) / (high - low)
    
        return np.clip(x, 0.0, 1.0)


    def _calculate_window_score(self, y: np.ndarray, sr: int, n_fft:int=2048, hop_length:int=512) -> float:
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
        rms_score = np.mean(self._normalize(rms_db))
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
        structure_score = 1.0 - np.mean(self._normalize(flatness))
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        onset = librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop_length)
        change_score = np.mean(self._normalize(onset))
        threshold = np.percentile(rms_db, 60)
        activity_ratio = np.mean(rms_db > threshold)

        # Combined score
        score = (
            0.45 * rms_score
            + 0.25 * structure_score
            + 0.15 * change_score
            + 0.15 * activity_ratio
        )

        return float(score)

 
    def _audio_preprocessing(self, audio_file: Path, target_sr: int = 32_000) -> Path | None:

        y, orig_sr = librosa.load(audio_file, sr=None, mono=True)
        duration = len(y) / orig_sr

        if y.ndim > 1:
            y = librosa.to_mono(y)

        if duration > self.window_sec:
            window_samples = int(self.window_sec * orig_sr)
            step_samples = int(orig_sr / 10)

            candidates = list()

            for start in range(
                0,
                len(y) - window_samples + 1,
                step_samples
            ):
        
                end = start + window_samples
                segment = y[start:end]
                score = self._calculate_window_score(segment, orig_sr)
        
                candidates.append({
                    "початковий_семпл": start,
                    "кінцевий_семпл": end,
                    "початковий_час": start / orig_sr,
                    "кінцевий_час": end / orig_sr,
                    "оцінка": score,
                })

            best = max(candidates, key=lambda x: x["оцінка"])
            
            start = best["початковий_семпл"]
            end = best["кінцевий_семпл"]
        
            y = y[start:end]

        if orig_sr != target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

        self.count += 1
        audio_path = self.root / 'audio' / f'{hex(self.count)[2:]}.wav'
        sf.write(audio_path, y, target_sr)
        return audio_path


    def main(self) -> None:
        audio_pattern = re.compile(r'\.(wav|mp3)$', re.IGNORECASE)
        
        with open(self.root / "audio-annotations.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label", "dataset"])

            for dataset_dir in ("Bird-MY10", "Bird-SEA10"):
                for audio_file in tqdm(
                    (self.root / dataset_dir / 'audio').glob("*/*"),
                    desc=dataset_dir,
                ):

                    if not audio_file.is_file(): continue
                    if not audio_pattern.search(audio_file.name): continue

                    audio_name = self._audio_preprocessing(audio_file)
                    if audio_name == None: continue

                    writer.writerow([audio_name.name, audio_file.parent.name, dataset_dir])
                    f.flush()


    def __call__(self, *args, **kwds) -> None:
        self.main()



class ImagePreprocessing:
    def __init__(self, root: Path = Path("dataset"), max_pixel: int = 768):
        
        self.max_pixel:int = max_pixel
        self.count:int = 0
        self.root:Path = root


    def _image_preprocessing(self, image_file: Path) -> Path | None:
        with Image.open(image_file) as img:
            width, height = img.size
            if width > self.max_pixel or height > self.max_pixel:
                scale = self.max_pixel / max(width, height)
                new_size = (round(width * scale), round(height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            image_path = self.root / 'image' / f'{hex(self.count)[2:]}.jpg'
            self.count += 1
            img.save(image_path)

            return image_path


    def main(self) -> None:
        image_pattern = re.compile(r'\.(jpg|png)$', re.IGNORECASE)
                
        with open(self.root / "image-annotations.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label", "dataset"])

            for dataset_dir in ("Bird-MY10", "Bird-SEA10"):
                for image_file in tqdm(
                    (self.root / dataset_dir / 'image').glob("*/*"),
                    desc=dataset_dir,
                    total=1000,
                ):

                    if not image_file.is_file(): continue
                    if not image_pattern.search(image_file.name): continue

                    image_name = self._image_preprocessing(image_file)
                    if image_name == None: continue

                    writer.writerow([image_name.name, image_file.parent.name, dataset_dir])
                    f.flush()


    def __call__(self, *args, **kwds) -> None:
        self.main()