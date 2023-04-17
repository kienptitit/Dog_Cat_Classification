from train import get_model
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import numpy as np


class Make_Test(nn.Module):
    def __init__(self, weight_path):
        super(Make_Test, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = get_model()
        self.model.load_state_dict(torch.load(weight_path))
        self.d = {
            0: 'cat',
            1: 'dog'
        }

    def forward(self, image_path):
        image = plt.imread(image_path)
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0)
        self.model.eval()

        out = 0 if self.model(image).squeeze(-1).item() < 0.5 else 1
        return self.d[out]


def visuzalize_loss():
    train_loss = np.load('Accuracy/test_losses.npy')
    train_acc = np.load('test1/test_accs.npy')
    plt.plot(train_loss)
    plt.plot(train_acc)
    plt.show()


def visualize(mk_test, data_path):
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 4, 4

    for i in range(1, rows * cols + 1):
        img = plt.imread(os.path.join("test1", data_path[i - 1]))
        label = mk_test(os.path.join("test1", data_path[i - 1]))
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(label)
        plt.axis(False)
    plt.show()


if __name__ == '__main__':
    folder = "test1"
    cnt = 0
    mk_test = Make_Test(r"E:\Python test Work\Hieu\Weight\model.pt")
    data_path = os.listdir(folder)[16:32]
    visualize(mk_test, data_path)
