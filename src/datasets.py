# Dependencies
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset


class BirdImageDataset(Dataset):
    def __init__(self, img_dir: Path, ann_file: str | Path, device):
        self.img_dir = img_dir

        self.ann = pd.read_csv(ann_file)
        self.ann["label"] = self.ann["label"].astype("category").cat.codes
        self.ann["dataset"] = self.ann["dataset"].astype("category").cat.codes



    def __getitem__(self, idx: int):
        pass

    def __len__(self):
        pass



if __name__ == "__main__":
    df = pd.read_csv("dataset/image-annotations.csv")
    df["label"] = df["label"].astype("category").cat.codes
    df["dataset"] = df["dataset"].astype("category").cat.codes
    print(df.head())
    print(df.info())