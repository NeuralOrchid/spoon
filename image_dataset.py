# Dependencies
from typing import Optional, Literal
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T


class BirdImageDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        ann_file: str | Path,
        train: bool = True,
        dst: Optional[Literal['Bird-MY10', 'Bird-SEA10']] = None,
    ):
        # Image folder
        self.img_dir = img_dir

        # Annotations file
        self.ann = pd.read_csv(ann_file)
        self.ann["label"] = self.ann["label"].astype("category").cat.codes

        if dst is not None:
            self.ann = self.ann[self.ann["dataset"] == dst]
            self.ann = self.ann.reset_index(drop=True)

        self.ann = self.ann.drop(columns="dataset")

        if train:
            self.transform = T.Compose([
                T.Resize(384),
                T.RandomResizedCrop(256, (0.8, 1.0), (0.95, 1.05)),
                T.RandomHorizontalFlip(),

                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1
                ),

                T.ToTensor(),

                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ])
            
        else:
            self.transform = T.Compose([
                T.Resize(256),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ])


    def __len__(self) -> int:
        return len(self.ann)


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
        

    def __getitem__(self, idx: int):
        image_path = self.img_dir / self.ann.loc[idx, "file"]
        label = int(self.ann.loc[idx, "label"])

        image = Image.open(image_path).convert("RGB")
        image = self._pad2square(image)

        image = self.transform(image)

        return image, label
    

if __name__ == "__main__":
    mydataset = BirdImageDataset(
        img_dir=Path('dataset/image'),
        ann_file=Path('dataset/image-annotations.csv'),
    )

    img, _ = mydataset[0]

    import torch
    import matplotlib.pyplot as plt

    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    img = img * std[:, None, None] + mean[:, None, None]
    img = img.permute(1, 2, 0)
    img = img.clamp(0, 1)

    plt.imshow(img)
    plt.axis("off")
    plt.show()