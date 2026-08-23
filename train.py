# Dependencies
from argparse import Namespace

from tqdm import trange

from . import (
    BirdImageDataset,
    BirdAudioDataset,
    UnpairedDataset,
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

        if args.image_mobilevitxxs_checkpoint.exists():
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
            preds, _ = self.model(inputs)
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
                preds, _ = self.model(inputs).argmax(dim=1)
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
            preds, _ = self.model(inputs)
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
                preds, _ = self.model(inputs).argmax(dim=1)
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


    def __call__(self, *args, **kwds):
        self.train()



class UnpairedTrainer:
    def __init__(self, args: Namespace):

        ## Load default Parameters
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Define Models: Image Modality Encoder
        self.audio_model = MobileViT_XXS(
            img_size = (128, 384),
            num_classes = args.num_classes,
            in_channels = 1,
        ).to(self.device)

        ## Define Models: Audio Modality Encoder
        self.image_model = MobileViT_XXS(
            img_size = args.image_size,
            num_classes = args.num_classes
        ).to(self.device)

        ## Define Models: Fusion Model
        self.fusion_model = nn.Sequential(
            nn.LayerNorm(640),
            nn.Linear(640, 256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, args.num_classes)
        ).to(self.device)

        ## Load Model Weights
        self._load_model_weights()

        ## Setup F1 Metric, Loss Function, Optimizer, and Scheduler
        self.f1_metric = MulticlassF1Score(args.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.fusion_model.parameters(),
            args.lr,
            momentum = 0.9,
            weight_decay = args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, args.epochs
        )

        ## Define Dataset: Image Modal Dataset
        image_dataset = BirdImageDataset(args.img_dir, args.img_ann, True)

        ## Define Dataset: Audio Modal Dataset
        audio_dataset = BirdAudioDataset(
            audio_dir=args.audio_dir,
            ann_file=args.audio_ann,
            sample_rate=args.sample_rate,
            audio_length=args.audio_length,
        )

        ## Define Datasets: Pairing Dataset
        unpaired_dataset = UnpairedDataset(
            image_dataset=image_dataset,
            audio_dataset=audio_dataset,
            samples_per_epoch=args.batch_size*8
        )

        ## Load Transform Functions
        self.audio_transform = audio_dataset.transform
        # self.image_transform = image_dataset.transform

        ## Define DataLoader
        self.data_loader = DataLoader(
            dataset=unpaired_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        

    def _save_model_weights(self) -> None:
        torch.save(
            self.fusion_model.state_dict(),
            self.args.fusion_model_checkpoint,
        )
    
    def _load_model_weights(self) -> None:
        if self.args.audio_mobilevitxxs_checkpoint.exists():
            state = torch.load(
                self.args.audio_mobilevitxxs_checkpoint, 
                map_location=self.device,
            )
            self.audio_model.load_state_dict(state)

        if self.args.image_mobilevitxxs_checkpoint.exists():
            state = torch.load(
                self.args.image_mobilevitxxs_checkpoint, 
                map_location=self.device,
            )
            self.image_model.load_state_dict(state)

        if self.args.fusion_model_checkpoint.exists():
            state = torch.load(
                self.args.fusion_model_checkpoint, 
                map_location=self.device,
            )
            self.fusion_model.load_state_dict(state)

    def _train_epoch(self) -> float:
        self.fusion_model.train()
        
        losses = list()

        for images, audios, labels in self.data_loader:
            # images = self.image_transform(images, train=False)
            images = images.to(self.device)
            audios = audios.to(self.device)
            audios = self.audio_transform(audios, train=True)
            labels = labels.to(self.device)

            _, image_features = self.image_model(images)
            _, audio_features = self.audio_model(audios)

            inputs = torch.cat(
                (
                    image_features,
                    audio_features
                ), dim=1
            )

            preds = self.fusion_model(inputs)

            loss = self.criterion(preds, labels)

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

        self.fusion_model.eval()

        for images, audios, labels in self.data_loader:
            # images = self.image_transform(images, train=False)
            images = images.to(self.device)
            audios = audios.to(self.device)
            audios = self.audio_transform(audios, train=False)
            labels = labels.to(self.device)

            with torch.no_grad():
                _, image_features = self.image_model(images)
                _, audio_features = self.audio_model(audios)

                inputs = torch.cat(
                    (
                        image_features,
                        audio_features
                    ), dim=1
                )

                preds = self.fusion_model(inputs).argmax(dim=1)

                all_preds.append(preds)
                all_targets.append(labels)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)


        return self.f1_metric(all_preds, all_targets).item()


    def train(self):
        self.audio_model.eval()
        self.image_model.eval()

        loop = trange(self.args.epochs +1)
        loop_postfix = {
            'loss': 0.0,
            'f1 score': 0.0,
        }

        for epoch in loop:
            loss = self._train_epoch()
            loop_postfix['loss'] = loss

            if epoch % 5 == 0:
                f1_score = self._val_epoch()
                loop_postfix['f1 score'] = f1_score

            loop.set_postfix(loop_postfix)
        self._save_model_weights()


    def __call__(self):
        # === Main ===
        self.train()


# Now the power is about to go out.
# Usually the neighborhood electricity is cut for two hours during the day.
# TODO: Remember I need to fix this part later. 
class PairedTrainer:
    def __init__(self, args: Namespace):

        ## Load default Parameters
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Define Models: Image Modality Encoder
        self.audio_model = MobileViT_XXS(
            img_size = (128, 384),
            num_classes = args.num_classes,
            in_channels = 1,
        ).to(self.device)

        ## Define Models: Audio Modality Encoder
        self.image_model = MobileViT_XXS(
            img_size = args.image_size,
            num_classes = args.num_classes
        ).to(self.device)

        ## Define Models: Fusion Model 
        self.fusion_model = nn.Sequential(
            nn.LayerNorm(640),
            nn.Linear(640, 256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, args.num_classes)
        ).to(self.device)

        ## Load Model Weights
        self._load_model_weights()

        ## Setup F1 Metric, Loss Function, Optimizer, and Scheduler
        self.f1_metric = MulticlassF1Score(args.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.fusion_model.parameters(), # FIXME
            args.lr,
            momentum = 0.9,
            weight_decay = args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, args.epochs
        )

        ## Define Dataset: Image Modal Dataset
        image_dataset = BirdImageDataset(args.img_dir, args.img_ann, True) # FIXME

        ## Define Dataset: Audio Modal Dataset
        audio_dataset = BirdAudioDataset( # FIXME
            audio_dir=args.audio_dir,
            ann_file=args.audio_ann,
            sample_rate=args.sample_rate,
            audio_length=args.audio_length,
        )

        ## Define Datasets: Pairing Dataset
        paired_dataset = PairedDataset( # FIXME
            image_dataset=image_dataset,
            audio_dataset=audio_dataset,
            samples_per_epoch=args.batch_size*8
        )

        ## Load transform function
        self.transform = audio_dataset.transform # FIXME

        ## Define DataLoader
        self.data_loader = DataLoader(
            dataset=paired_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
        

    def _save_model_weights(self) -> None:
        torch.save(
            self.image_model.state_dict(),
            self.args.image_mobilevitxxs_checkpoint,
        )
        torch.save(
            self.audio_model.state_dict(),
            self.args.audio_mobilevitxxs_checkpoint,
        )
        torch.save(
            self.fusion_model.state_dict(),
            self.args.fusion_model_checkpoint,
        )
    
    def _load_model_weights(self) -> None:
        if self.args.image_mobilevitxxs_checkpoint.exists():
            state = torch.load(
                self.args.image_mobilevitxxs_checkpoint, 
                map_location=self.device,
            )
            self.image_model.load_state_dict(state)

        if self.args.audio_mobilevitxxs_checkpoint.exists():
            state = torch.load(
                self.args.audio_mobilevitxxs_checkpoint, 
                map_location=self.device,
            )
            self.audio_model.load_state_dict(state)

        if self.args.fusion_model_checkpoint.exists():
            state = torch.load(
                self.args.fusion_model_checkpoint, 
                map_location=self.device,
            )
            self.fusion_model.load_state_dict(state)

    def _train_epoch(self) -> float:
        self.fusion_model.train()
        
        losses = list()

        for images, audios, labels in self.data_loader:
            # images = self.image_transform(images, train=False)
            images = images.to(self.device)
            audios = audios.to(self.device)
            audios = self.transform(audios, train=True)
            labels = labels.to(self.device)

            _, image_features = self.image_model(images)
            _, audio_features = self.audio_model(audios)

            inputs = torch.cat(
                (
                    image_features,
                    audio_features
                ), dim=1
            )

            preds = self.fusion_model(inputs)

            loss = self.criterion(preds, labels)

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

        self.image_model.eval()
        self.audio_model.eval()
        self.fusion_model.eval()

        for images, audios, labels in self.data_loader:
            images = images.to(self.device)
            audios = audios.to(self.device)
            audios = self.transform(audios, train=False)
            labels = labels.to(self.device)

            with torch.no_grad():
                _, image_features = self.image_model(images)
                _, audio_features = self.audio_model(audios)

                inputs = torch.cat(
                    (
                        image_features,
                        audio_features
                    ), dim=1
                )

                preds = self.fusion_model(inputs).argmax(dim=1)

                all_preds.append(preds)
                all_targets.append(labels)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)


        return self.f1_metric(all_preds, all_targets).item()


    def __call__(self):
        loop = trange(self.args.epochs +1)
        loop_postfix = {
            'loss': 0.0,
            'f1 score': 0.0,
        }

        for epoch in loop:
            loss = self._train_epoch()
            loop_postfix['loss'] = loss

            if epoch % 5 == 0:
                f1_score = self._val_epoch()
                loop_postfix['f1 score'] = f1_score

            loop.set_postfix(loop_postfix)
        # self._save_model_weights() # FIXME: uncomment if necessary