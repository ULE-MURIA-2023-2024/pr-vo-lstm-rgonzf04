
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import VisualOdometryDataset
from model import VisualOdometryModel
from params import *


# Create the visual odometry model
model = VisualOdometryModel(hidden_size, num_layers)

transform = T.Compose([
    T.ToTensor(),
    model.resnet_transforms()
])


# TODO: Load dataset
train_loader = DataLoader(dataset=VisualOdometryDataset(
    dataset_path="dataset/val",
    transform=transform,
    sequence_length=sequence_length,
    validation=True
), batch_size=batch_size)


# val
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
model.load_state_dict(torch.load("vo.pt"))
model.eval()

validation_string = ""
position = [0.0] * 7

with torch.no_grad():
    for images, labels, timestamp in tqdm(train_loader, f"Validating:"):

        images = images.to(device)
        labels = labels.to(device)

        target = model(images).cpu().numpy().tolist()[0]

        # TODO: add the results into the validation_string

        for i in range(7):
            position[i] += target[i]

        validation_string += f"{timestamp[0]} {position[0]} {position[1]} {position[2]} {position[3]} {position[4]} {position[5]} {position[6]}\n"


f = open("validation.txt", "a")
f.write(validation_string)
f.close()
