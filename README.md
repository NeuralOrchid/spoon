# Cross-Dataset Generalization of Multimodal Bird Species Recognition Without Target-Domain Adaptation

PyTorch code for bird-species recognition from images, audio, and paired multimodal inputs using MobileViT models.

## Code map

- `preprocessing.py` — prepares image/audio files and annotation CSVs.
- `image_dataset.py`, `audio_dataset.py` — modality-specific datasets and transforms.
- `paired_dataset.py` — paired image/audio samples.
- `mobilevit.py` — MobileViT models and attention extraction.
- `train.py` — unimodal, multimodal, and late-fusion training/evaluation.