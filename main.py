import data
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from config import cfg
from model import HourglassModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = .001
    annotations_path = "filtered_annotations/single_person_keypoints_train2017.txt"
    images_path = "train2017/"

    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = data.CocoDataloader(annotations_path, images_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=data.collate)
    model = HourglassModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)

        print(inputs.shape)

        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        print(f"Training loss: {loss.item()}")

if __name__ == "__main__":
    main()