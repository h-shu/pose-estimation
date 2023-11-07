import torch.utils.data as data
import torch
import cv2
import numpy as np

from torchvision import transforms
from config import cfg

class CocoDataloader(data.Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.image_dir = image_dir
        self.images = []
        self.labels = []

        # Read in images and annotations for each image.
        idx = 0
        for line in open(annotations_file, "r"):
            if idx % 2 == 0:
                self.images.append(line.strip())
            else:
                label = [int(x) for x in line.strip().split(",")]
                self.labels.append(label)
            idx += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load in image and label.
        img_path = self.image_dir + self.images[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)

        # Convert image and label to tensors (NCHW) and apply transforms if available.
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).to(torch.float)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = torch.tensor(label, dtype=torch.float)

        # Return input and label as tensors.
        return image, label
    
def show_img_label_after_resizing(resized_image, resized_label):
    # Used for debugging purposes.
    show_img = resized_image.permute(1,2,0).cpu().detach().numpy().copy()
    show_label = resized_label.cpu().detach().numpy()
    for i in range(0, len(show_label), 3):
        x = int(show_label[i])
        y = int(show_label[i+1])
        cv2.circle(show_img, (x, y), 2, (0, 0, 255), -1)        
    cv2.imshow("Image", show_img)
    cv2.waitKey(0)

def collate(batch):
    # Get max height and width to resize images.
    max_height = cfg["image_height"]
    max_width = cfg["image_width"]

    batch_images = []
    batch_labels = []

    # Resize images to same sizes.
    for image, label in batch:
        image_height = image.shape[1]
        image_width = image.shape[2]
        resized_image = transforms.Resize((max_height, max_width), antialias=True)(image)
        resized_label = torch.zeros(51)

        # (x, y, visible)
        for idx in range(0, len(label), 3):
            resized_label[idx] = (label[idx] * max_width / image_width)
            resized_label[idx+1] = (label[idx+1] * max_height / image_height)
            resized_label[idx+2] = (label[idx+2])

        batch_images.append(resized_image)
        batch_labels.append(resized_label)

    batch_images = torch.stack(batch_images, dim=0)
    batch_labels = torch.stack(batch_labels, dim=0)

    return batch_images, batch_labels