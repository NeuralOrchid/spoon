# Dependencies
from typing import Optional, Literal
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset


class BirdImageDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        ann_file: str | Path,
        transformation,
        dst: Optional[Literal['Bird-MY10', 'Bird-SEA10']] = None,
        device: Optional[str]='cpu',
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

        # Device cuda/cpu
        self.device = device
        self.transformation = transformation.to(self.device)


    def __len__(self) -> int:
        return len(self.ann)


    def __getitem__(self, idx: int):
        pass



if __name__ == "__main__":
    mydataset = BirdImageDataset(
        img_dir=Path('dataset/image'),
        ann_file=Path('dataset/image-annotations.csv'),
    )

    print(len(mydataset))