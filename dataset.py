
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Callable


class VisualOdometryDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        transform: Callable,
        sequence_length: int,
        validation: bool = False
    ) -> None:

        self.sequences = []

        directories = [d for d in os.listdir(
            dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        for subdir in directories:

            aux_path = f"{dataset_path}/{subdir}"

            # read data
            rgb_paths = self.read_images_paths(aux_path)

            if not validation:
                ground_truth_data = self.read_ground_truth(aux_path)
                interpolated_ground_truth = self.interpolate_ground_truth(
                    rgb_paths, ground_truth_data)
                
            # TODO: create sequences

            
    
            for i in range(1, len(rgb_paths), 1):

                if not validation:
                    gt = [
                        interpolated_ground_truth[i][1][0] - interpolated_ground_truth[i-1][1][0],
                        interpolated_ground_truth[i][1][1] - interpolated_ground_truth[i-1][1][1],
                        interpolated_ground_truth[i][1][2] - interpolated_ground_truth[i-1][1][2],
                        interpolated_ground_truth[i][1][3] - interpolated_ground_truth[i-1][1][3],
                        interpolated_ground_truth[i][1][4] - interpolated_ground_truth[i-1][1][4],
                        interpolated_ground_truth[i][1][5] - interpolated_ground_truth[i-1][1][5],
                        interpolated_ground_truth[i][1][6] - interpolated_ground_truth[i-1][1][6]]
                     
                if validation:
                    gt = []

                self.sequences.append(
                    [
                     rgb_paths[i][0], # timestamp
                     rgb_paths[i-1][1], # image i-1
                     rgb_paths[i][1], # image i
                     gt # ground truth
                    ]
                )

        self.transform = transform
        self.sequence_length = sequence_length
        self.validation = validation

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.TensorType:

        # Load sequence of images
        sequence_images = []
        ground_truth_pos = []
        timestampt = 0

        # TODO: return the next sequence

        sequence = self.sequences[idx]

        timestampt = sequence[0]

        rgb_img1 = cv2.imread(sequence[1])
        rgb_img1 = cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2RGB)
        rgb_img1 = self.transform(rgb_img1)

        rgb_img2 = cv2.imread(sequence[2])
        rgb_img2 = cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2RGB)
        rgb_img2 = self.transform(rgb_img2)

        sequence_images = torch.stack([rgb_img1, rgb_img2])

        # Get ground truth translation

        ground_truth_pos = torch.FloatTensor(sequence[3])

        # if not self.validation:
        #     ground_truth_pos = torch.FloatTensor([
        #         sequence[3][self.sequence_length - 1][0] -
        #         sequence[3][self.sequence_length - 2][0],

        #         sequence[3][self.sequence_length - 1][1] -
        #         sequence[3][self.sequence_length - 2][1],

        #         sequence[3][self.sequence_length - 1][2] -
        #         sequence[3][self.sequence_length - 2][2],

        #         sequence[3][self.sequence_length - 1][3] -
        #         sequence[3][self.sequence_length - 2][3],

        #         sequence[3][self.sequence_length - 1][4] -
        #         sequence[3][self.sequence_length - 2][4],

        #         sequence[3][self.sequence_length - 1][5] -
        #         sequence[3][self.sequence_length - 2][5],

        #         sequence[3][self.sequence_length - 1][6] -
        #         sequence[3][self.sequence_length - 2][6]
        #     ])
        # else:
        #     ground_truth_pos = torch.FloatTensor([])

        # Get timestamp
        timestampt = self.sequences[idx][0]

        return sequence_images, ground_truth_pos, timestampt

    def read_images_paths(self, dataset_path: str) -> Tuple[float, str]:

        paths = []

        with open(f"{dataset_path}/rgb.txt", "r") as file:
            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                image_path = f"{dataset_path}/{line[1]}"

                paths.append((timestamp, image_path))

        return paths

    def read_ground_truth(self, dataset_path: str) -> Tuple[float, Tuple[float]]:

        ground_truth_data = []

        with open(f"{dataset_path}/groundtruth.txt", "r") as file:

            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                position = list(map(float, line[1:]))
                ground_truth_data.append((timestamp, position))

        return ground_truth_data

    def interpolate_ground_truth(
            self,
            rgb_paths: Tuple[float, str],
            ground_truth_data: Tuple[float, Tuple[float]]
    ) -> Tuple[float, Tuple[float]]:

        rgb_timestamps = [rgb_path[0] for rgb_path in rgb_paths]
        ground_truth_timestamps = [item[0] for item in ground_truth_data]

        # Interpolate ground truth positions for each RGB image timestamp
        interpolated_ground_truth = []

        for rgb_timestamp in rgb_timestamps:

            nearest_idx = np.argmin(
                np.abs(np.array(ground_truth_timestamps) - rgb_timestamp))

            interpolated_position = ground_truth_data[nearest_idx]
            interpolated_ground_truth.append(interpolated_position)

        return interpolated_ground_truth
