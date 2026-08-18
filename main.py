# Dependencies
from argparse import Namespace
from pathlib import Path
from .src import ImageTrainer

DEFAULTS = Namespace(
    image_size      = 256,
    num_classes     = 10,
    lr              = 4.5e-2,
    epochs          = 90,
    batch_size      = 71,
    weight_decay    = 5e-5,
    img_dir         = Path('dataset/image'),
    audio_dir       = Path('dataset/audio'),
    img_ann         = Path('dataset/image-annotations.csv'),
    audio_ann       = Path('dataset/audio-annotations.csv'),
    mobilevitxxs_checkpoint = 'drive/MyDrive/MobileViT_XXS.pth'
)


def main() -> None:
    trainer = ImageTrainer(DEFAULTS)
    trainer.train()
    


if __name__ == "__main__":
    main()