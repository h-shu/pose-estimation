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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=data.collate)

    # Set up model.
    model = HourglassModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # Train the model.
    epochs = list(range(cfg["epochs"]))
    iterations = []
    losses = []

    true_iteration = 0
    for epoch in epochs:
        model.train()
        iteration = 0

        for inputs, labels in trainloader:
            # Init time.
            t0 = time.time()

            # Load in inputs and labels -> predict.
            inputs, labels = inputs.to(device), labels.to(device)
            pred = model(inputs)

            # Backprop.
            optimizer.zero_grad()
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            # Get estimated time remaining.
            t1 = time.time()
            iteration += 1
            true_iteration += 1
            epochs_left = epochs[-1] - epoch
            inputs_left = len(trainloader) - iteration
            eta = (t1-t0) * inputs_left + (t1 - t0) * len(trainloader) * epochs_left

            # Log info.
            print(f"Epoch: {epoch+1}/{epochs[-1]+1} || Iteration: {iteration}/{len(trainloader)} || MSE Loss: {loss.item()} || ETA: {str(datetime.timedelta(seconds=int(eta)))}")
            iterations.append(true_iteration)
            losses.append(loss.item())

    # Save plot.
    plt.plot(iterations, losses)
    plt.title("MSE Loss vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.savefig(cfg["plots_path"] + "plot.png", dpi=300)

    # Save the weights.
    named_date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    torch.save(model.state_dict(), cfg["saved_path"] + named_date)

if __name__ == "__main__":
    main()