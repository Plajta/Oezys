#
# A sample inference script to test out models
#

import torch
import cv2
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.nn_model import ResNet18Model
from src.models.data import load_config
from os.path import join


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def inference_setup(checkpoint_path, config_path, image_folder):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_config = load_config(config_path)

    # 2. Load Model
    model = ResNet18Model.load_from_checkpoint(checkpoint_path, config=resnet_config)
    model.to(device)
    model.eval()

    # 3. Transform (Must match validation exactly)
    transform = A.Compose([
        A.Resize(224, 224), # Standard ResNet size
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 4. Find BMPs
    img_paths = glob.glob(join(image_folder, "*.bmp"))

    with torch.no_grad():
        for path in img_paths:
            label_path = path.replace(".bmp", ".txt").replace("imgs", "labels")

            label = read_file(label_path)

            img_bgr = cv2.imread(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Preprocess
            tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

            # Predict
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item() + 1 # +1 to return to 1-5 range

            print(f"Image: {path.split('/')[-1]} | Predicted Class: {pred_class} | Actual Class: {label} | Confidence: {probs.max():.2f}")
