import data
import datetime
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from config import cfg
from model import HourglassModel

def main():
    # Find the device to sent to.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in data.
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = data.CocoDataloader(cfg["annotations_path"], cfg["images_path"], transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=data.collate)

    # Set up model.
    model = HourglassModel().to(device)
    weights = torch.load(cfg["saved_path"] + "2023_11_08_09_35_37")
    model.load_state_dict(weights)
    model.eval()

    # Predict using the model.
    print("Predicting.")

    for input, _ in trainloader:
        # Load in inputs and labels -> predict.
        input = input.to(device)
        pred = model(input)

        input = input[0, :, :, :]
        print(f"Input shape: {input.shape}")
        print(f"Pred shape: {pred.shape}")
        print(f"Pred: {pred}")

        data.show_img_label_after_resizing(input, pred)



if __name__ == "__main__":
    main()