# Local libs import
import trainer
from loader import load_config
from nn_model import ResNet18Model


if __name__ == "__main__":
    resnet_config = load_config("./config/nn_models/resnet.yaml")
    ResNet18Model(resnet_config)