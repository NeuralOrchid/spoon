# Dependencies
from argparse import Namespace

from tqdm import trange

from . import (
    BirdImageDataset,
    BirdAudioDataset,
    MobileViT_XXS,
)

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassF1Score



class ImageTrainer:
    def __init__(self, args: Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model / network
        self.model = MobileViT_XXS(
            img_size = args.image_size,
            num_classes = args.num_classes
        ).to(self.device)

        # Loss function & Optimizer
        self.f1_metric = MulticlassF1Score(args.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            args.lr,
            momentum = 0.9,
            weight_decay = args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, args.epochs
        )

        # DataLoaders
        self.train_loader = DataLoader(
            BirdImageDataset(args.img_dir, args.img_ann, True),
            batch_size = args.batch_size, shuffle = True,
            num_workers = 1,  pin_memory = True
        )
        self.val_loader = DataLoader(
            BirdImageDataset(args.img_dir, args.img_ann, False),
            batch_size = args.batch_size, shuffle = True,
            num_workers = 1,  pin_memory = True
        )


    def _save_model_weights(self) -> None:
        torch.save(
            self.model.state_dict(),
            self.args.image_mobilevitxxs_checkpoint,
        )

    def _load_model_weights(self) -> None:
        state = torch.load(
            self.args.image_mobilevitxxs_checkpoint, 
            map_location=self.device,
        )
        self.model.load_state_dict(state)


    def _train_epoch(self) -> float:
        self.model.train()

        losses = list()

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            preds = self.model(inputs)
            loss = self.criterion(preds, targets)

            losses.append(loss.item())

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

        return sum(losses) / len(losses)


    def _val_epoch(self) -> float:
        all_preds = []
        all_targets = []

        self.model.train(False)

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                preds = self.model(inputs).argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(targets)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)


        return self.f1_metric(all_preds, all_targets).item()


    def train(self):
        loop = trange(self.args.epochs +1)
        loop_postfix = {'loss': 0.0, 'f1 score': 0.0}

        for epoch in loop:
            loss = self._train_epoch()
            loop_postfix['loss'] = loss

            if epoch % 5 == 0:
                f1_score = self._val_epoch()
                loop_postfix['f1 score'] = f1_score

            loop.set_postfix(loop_postfix)

        self._save_model_weights()


class AudioTrainer:
    def __init__(self, args: Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model / network
        self.model = MobileViT_XXS(
            img_size = (128, 384),
            num_classes = args.num_classes,
            in_channels = 1,
        ).to(self.device)

        if args.audio_mobilevitxxs_checkpoint.exists():
            self._load_model_weights()

        # Loss function & Optimizer
        self.f1_metric = MulticlassF1Score(args.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            args.lr,
            momentum = 0.9,
            weight_decay = args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, args.epochs
        )

        # DataLoaders
        audio_dataset = BirdAudioDataset(
            audio_dir=args.audio_dir,
            ann_file=args.audio_ann,
            sample_rate=args.sample_rate,
            audio_length=args.audio_length,
        )

        self.transform = audio_dataset.transform

        self.data_loader = DataLoader(
            dataset=audio_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )


    def _save_model_weights(self) -> None:
        torch.save(
            self.model.state_dict(),
            self.args.audio_mobilevitxxs_checkpoint,
        )

    def _load_model_weights(self) -> None:
        state = torch.load(
            self.args.audio_mobilevitxxs_checkpoint, 
            map_location=self.device,
        )
        self.model.load_state_dict(state)


    def _train_epoch(self) -> float:
        self.model.train()

        losses = list()

        for inputs, targets in self.data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            inputs = self.transform(inputs, train=True)

            # Forward pass
            preds = self.model(inputs)
            loss = self.criterion(preds, targets)

            losses.append(loss.item())

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

        return sum(losses) / len(losses)


    def _val_epoch(self) -> float:
        all_preds = []
        all_targets = []

        self.model.train(False)

        for inputs, targets in self.data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            inputs = self.transform(inputs, train=False)

            with torch.no_grad():
                preds = self.model(inputs).argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(targets)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)


        return self.f1_metric(all_preds, all_targets).item()


    def train(self):
        loop = trange(self.args.epochs +1)
        loop_postfix = {'loss': 0.0, 'f1 score': 0.0}

        for epoch in loop:
            loss = self._train_epoch()
            loop_postfix['loss'] = loss

            if epoch % 1 == 0:
                f1_score = self._val_epoch()
                loop_postfix['f1 score'] = f1_score

            loop.set_postfix(loop_postfix)
        self._save_model_weights()


    def __call__(self, *args, **kwds):
        self.train()