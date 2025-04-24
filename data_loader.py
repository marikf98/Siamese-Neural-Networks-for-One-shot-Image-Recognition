import torch
import cv2

import os
from collections import defaultdict
import random

class data_loader(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.people = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]
        self.image_paths = []
        self.labels = []

        for i, person_name in enumerate(self.people):
            person_dir = os.path.join(self.data_dir, person_name)

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(i)

        self.label_to_images = defaultdict(list)
        for img_path, label in zip(self.image_paths, self.labels):
            self.label_to_images[label].append(img_path)
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path1 = self.image_paths[idx]
        label1 = self.labels[idx]
        same_class = random.randint(0, 1)  # 1 for same, 0 for different

        if same_class:
            possible_imgs = [p for p in self.label_to_images[label1] if p != img_path1]
            img_path2 = random.choice(possible_imgs)

        else:
            label2 = random.choice([l for l in self.label_to_images.keys() if l != label1])
            img_path2 = random.choice(self.label_to_images[label2])


        image1 = cv2.imread(img_path1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = cv2.resize(image1, (105, 105)) # size like in the paper
        image1 = image1.astype('float32') / 255.0 # normalize to [0, 1]
        image1 = torch.from_numpy(image1).permute(2, 0, 1) # convert to tensor and change the order of the dimensions from HWC to CHW

        image2 = cv2.imread(img_path2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = cv2.resize(image2, (105, 105))
        image2 = image2.astype('float32') / 255.0
        image2 = torch.from_numpy(image2).permute(2, 0, 1) # convert to tensor and change the order of the dimensions from HWC to CHW

        return image1, image2, torch.tensor(int(same_class), dtype=torch.float32) # torch.tensor([int(same_class)], dtype=torch.float32) is the label for the images in a tensor format 1 same person, 0 different person