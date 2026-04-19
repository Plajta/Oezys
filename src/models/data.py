#
# All the loading utilities + Dataset + Dataloader classes
#

import yaml
import albumentations as A
import numpy as np
import cv2
import torch
from os import listdir
from os.path import join

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        return config


class DataInspector:
    def __init__(self, abs_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_path = join(abs_path, "src/models/config")

        # Configs load
        self.data_config = load_config(join(config_path, "data.yaml"))
        self.augmentations_config = self.data_config["augmentations"]
        self.dataloader_config = self.data_config["dataloader"]

        # Dataset path
        label_dir_path = join(abs_path, "data/clean/labels")
        img_dir_path = join(abs_path, "data/clean/imgs")

        aggregator = TearAggregator(
            self.data_config,
            label_dir_path,
            img_dir_path
        )

        # Stratified splitting + automatic augmentation on-the-grab
        labels, images = aggregator.extract()

        self.dataset = TearDataset(labels, images, self.data_config, "train", device)
        self.dataloader = TearDataloader(self.dataset, self.dataloader_config)
        self.class_names = ["1", "2", "3", "4", "5"]

    def denormalize(self, tensor, mean, std):
        """Undo normalization so the image looks right in cv2."""
        t = tensor.clone()
        for ch, m, s in zip(t, mean, std):
            ch.mul_(s).add_(m)
        return t

    def tensor_to_bgr(self, tensor, mean=None, std=None):
        """Convert CHW float tensor → HWC uint8 BGR for cv2."""
        if mean and std:
            tensor = self.denormalize(tensor, mean, std)
        img = tensor.cpu().permute(1, 2, 0).numpy()          # CHW → HWC
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)     # RGB → BGR for cv2
        return img

    def debug_dataloader(self, n_batches=3):
        print(f"Dataset size : {len(self.dataset)}")
        print(f"Num batches  : {len(self.dataloader)}")
        print("-" * 40)

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx >= n_batches:
                break

            print(f"\nBatch {batch_idx}")
            print(f"  images shape : {images.shape}  dtype: {images.dtype}")
            print(f"  labels       : {labels.tolist()}")

            # Print per-sample label
            for i, label in enumerate(labels):
                label_int = label.item()
                label_str = str(label_int)
                print(f"  sample {i:>2d} — label: {label_int} ({label_str})")

            # Show images with cv2
            for i, img_tensor in enumerate(images):
                img = self.tensor_to_bgr(
                    img_tensor,
                    mean=self.augmentations_config["normalize"]["mean_vals"],
                    std=self.augmentations_config["normalize"]["std_vals"]
                )

                label_int = labels[i].item()
                label_str = str(label_int)
                window_title = f"Batch {batch_idx} | Sample {i} | {label_str}"

                cv2.imshow(window_title, img)
                key = cv2.waitKey(0)  # press any key for next image
                cv2.destroyAllWindows()

                if key == ord("q"):  # press Q to quit entirely
                    print("Quitting debug.")
                    return

        print("\nDone.")


def stratified_split(labels, images, config, device, model_type = "resnet", experts=None):
    train_labels, temp_labels, train_images, temp_images = train_test_split(
        labels, images,
        test_size=config["test_size"],
        stratify=labels
    )

    val_labels, test_labels, val_images, test_images = train_test_split(
        temp_labels, temp_images,
        test_size=0.5,
        stratify=temp_labels
    )

    if model_type == "moe" and experts is not None:
        # Switch to Linear MoE dataset, which have aggressive augmentations
        rf_expert, resnet_expert = experts
        train_dataset = MoEDataset(train_labels, train_images, rf_expert, resnet_expert, config, "train")
        test_dataset = MoEDataset(val_labels, val_images, rf_expert, resnet_expert, config, "val")
        val_dataset = MoEDataset(test_labels, test_images, rf_expert, resnet_expert, config, "test")
    else:
        train_dataset = TearDataset(train_labels, train_images, config, "train", device)
        test_dataset = TearDataset(test_labels, test_images, config, "test", device)
        val_dataset = TearDataset(val_labels, val_images, config, "val", device)

    return train_dataset, test_dataset, val_dataset


# Aggregator class to get all paths and labels for training
class TearAggregator:
    def __init__(self, config, labels_path, imgs_path):
        self.config = config
        self.labels_path = labels_path
        self.imgs_path = imgs_path

        # Images
        images_list = listdir(self.imgs_path)
        self.images = [join(imgs_path, f) for f in images_list]

        # Label setup
        self.labels = [0] * len(self.images)
        for i, image_path in enumerate(self.images):
            label_path = image_path.replace("imgs", "labels").replace(".bmp", ".txt")
            self.labels[i] = int(self.read_resource(label_path))

    def sort_by_idx(self, str_name: str):
        return int(str_name.split('.')[0])

    def read_resource(self, label_path):
        with open(label_path, "r") as f:
            return f.read()

    def extract(self):
        return self.labels, self.images


class MoEDataset(Dataset):
    def __init__(self, labels, images, rf_expert, resnet_expert, config, mode="train"):
        self.labels = labels
        self.images = images
        self.rf_expert = rf_expert     # cl_inference.TearClassifier
        self.resnet_expert = resnet_expert # nn_inference.ResNetInference
        self.mode = mode

        # Tvůj Agresivní setup z yaml/configu
        self.transform = A.Compose([
            A.Rotate(limit=90, p=0.8),
            A.RandomBrightnessContrast(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5), # Tohle RF fakt nesnáší
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]) if mode == "train" else None

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Label 1-5 -> 0-4
        label = torch.tensor(int(self.labels[idx]) - 1, dtype=torch.long)

        if self.transform:
            image = self.transform(image=image)["image"]

        # 1. RF Expert (potřebuje cestu nebo pole, tvůj cl_inference bere cestu)
        # Hack: Protože cl_inference bere cestu, RF neuvidí tu augmentaci. 
        # Fix: Pojďme vytáhnout probas přímo z tvých expertů.
        rf_probas = self.rf_expert.predict_proba(img_path) # list [5]
        
        # 2. ResNet Expert
        with torch.no_grad():
            # Tady pozor: resnet_expert.run vrací list listů, bereme první
            resnet_probas = self.resnet_expert.run(img_path)[0] # list [5]

        # Spojíme do vektoru délky 10
        combined = torch.tensor(list(rf_probas) + resnet_probas, dtype=torch.float32)
        
        return combined, label


class AggresiveTearDataset(Dataset):
    def __init__(self, labels, images, config, dataset_type, device):
        self.labels = labels
        self.images = images
        self.aug_config = config["augmentations"]
        self.dataset_type = dataset_type
        self.device = device
        self.num_classes = config["num_classes"]

        size = self.aug_config["resize"]["size"]
        hflip_perc = self.aug_config["hflip"]["perc"]
        rotate_perc = self.aug_config["rotate"]["perc"]
        rotate_limit = self.aug_config["rotate"]["limit"]
        contrast_perc = self.aug_config["contrast"]["perc"]
        gauss_noise_perc = self.aug_config["gauss_noise"]["perc"]
        norm_mean_vals = self.aug_config["normalize"]["mean_vals"]
        norm_std_vals = self.aug_config["normalize"]["std_vals"]

        # Albumentation transform definitions
        self.train_transform = A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=hflip_perc),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Rotate(limit=rotate_limit, p=rotate_perc),
            A.RandomBrightnessContrast(p=contrast_perc),
            A.GaussNoise(p=gauss_noise_perc),
            A.Normalize(mean=norm_mean_vals, std=norm_std_vals),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=norm_mean_vals, std=norm_std_vals),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)
        label_idx = torch.tensor(int(self.labels[idx]) - 1, dtype=torch.long)

        return image, label_idx

    def transform(self, img):
        if self.dataset_type == "train":
            return self.train_transform(image=img)["image"]
        else:
            # Is val or test
            return self.val_transform(image=img)["image"]


class TearDataset(Dataset):
    def __init__(self, labels, images, config, dataset_type, device):
        self.labels = labels
        self.images = images
        self.aug_config = config["augmentations"]
        self.dataset_type = dataset_type
        self.device = device
        self.num_classes = config["num_classes"]

        size = self.aug_config["resize"]["size"]
        hflip_perc = self.aug_config["hflip"]["perc"]
        rotate_perc = self.aug_config["rotate"]["perc"]
        rotate_limit = self.aug_config["rotate"]["limit"]
        contrast_perc = self.aug_config["contrast"]["perc"]
        gauss_noise_perc = self.aug_config["gauss_noise"]["perc"]
        norm_mean_vals = self.aug_config["normalize"]["mean_vals"]
        norm_std_vals = self.aug_config["normalize"]["std_vals"]

        # Albumentation transform definitions
        self.train_transform = A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=hflip_perc),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Rotate(limit=rotate_limit, p=rotate_perc),
            A.RandomBrightnessContrast(p=contrast_perc),
            A.GaussNoise(p=gauss_noise_perc),
            A.Normalize(mean=norm_mean_vals, std=norm_std_vals),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=norm_mean_vals, std=norm_std_vals),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)
        label_idx = torch.tensor(int(self.labels[idx]) - 1, dtype=torch.long)

        return image, label_idx

    def transform(self, img):
        if self.dataset_type == "train":
            return self.train_transform(image=img)["image"]
        else:
            # Is val or test
            return self.val_transform(image=img)["image"]


class TearDataloader:
    def __init__(self, dataset: TearDataset, dataloader_config):
        self.loader = DataLoader(
            dataset,
            batch_size=dataloader_config["batch_size"],
            shuffle=dataloader_config["shuffle"],
            num_workers=dataloader_config["num_workers"]
        )

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)
