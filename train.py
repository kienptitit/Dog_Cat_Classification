from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import numpy as np


class Mydata(Dataset):
    def __init__(self, data_path, transform, label):
        super(Mydata, self).__init__()
        self.transform = transform
        self.label = label
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img = Image.fromarray(plt.imread(self.data_path[idx]))
        img = self.transform(img)
        label = self.label[idx]
        return img, label


def visualize(labels, data_path):
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 4, 4

    d_label = {
        0: 'Cat',
        1: 'Dog',
    }

    for i in range(1, rows * cols + 1):
        random_idx = random.randint(0, len(data_path) - 1)
        img = plt.imread(data_path[random_idx])
        label = d_label[labels[random_idx]]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(label)
        plt.axis(False)
    plt.show()


def get_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    return model


def get_data():
    train_path = [os.path.join("train", name) for name in os.listdir("train")]
    cat_train_path = [x for x in train_path if 'cat' in x]
    dog_train_path = [x for x in train_path if 'dog' in x]
    train_data_path = cat_train_path[:int(len(cat_train_path) * 0.8)] + dog_train_path[:int(len(dog_train_path) * 0.8)]
    val_data_path = cat_train_path[int(len(cat_train_path) * 0.8):] + dog_train_path[int(len(dog_train_path) * 0.8):]
    random.shuffle(train_data_path)
    random.shuffle(val_data_path)
    labels_train = [0 if 'cat' in name else 1 for name in train_data_path]
    labels_val = [0 if 'cat' in name else 1 for name in val_data_path]
    return train_data_path, val_data_path, labels_train, labels_val


def get_dataloader():
    train_data_path, val_data_path, labels_train, labels_val = get_data()
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = Mydata(train_data_path, transform=transform, label=labels_train)
    val_data = Mydata(val_data_path, transform=transform, label=labels_val)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=16, num_workers=2)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=16, num_workers=2)
    return train_loader, val_loader


def train_step(model,
               dataloader,
               loss_fn,
               optimizer,
               device):
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        y_pred = y_pred.squeeze(-1)

        loss = loss_fn(y_pred, y.float())
        train_loss += loss.item()
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.tensor([0 if x < 0.5 else 1 for x in y_pred]).to(device)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        if batch % 100 == 0:
            print(y_pred.squeeze(-1), y)
            print()
            print(
                "Batch {} Loss {:.4f} Train Acc {}".format(batch, loss, (y_pred_class == y).sum().item() / len(y_pred)))
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(model,
             dataloader,
             loss_fn: torch.nn.Module,
             device: torch.device):
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            test_pred_logits = test_pred_logits.squeeze(-1)
            loss = loss_fn(test_pred_logits, y.float())
            test_loss += loss.item()

            test_pred_labels = torch.tensor([0 if x < 0.5 else 1 for x in test_pred_logits]).to(device)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


if __name__ == '__main__':
    # Config
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoches = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=3e-5)
    loss = nn.BCELoss()
    model = model.to(device)
    # Data
    train_loader, val_loader = get_dataloader()
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for epoch in range(num_epoches):
        train_loss, train_acc = train_step(model, train_loader, loss, optimizer, device)
        print("Epoch {} Train_loss {} Train_acc {}".format(epoch, train_loss, train_acc))
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_loss, test_acc = val_step(model, val_loader, loss, device)
        print("Epoch {} Test_loss {} Test_acc {}".format(epoch, test_loss, test_acc))
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print("Done Epoch {}".format(epoch))
        # print("________________________________________________________________")
    torch.save(model.state_dict(), "Weight/model.pt")
    np.save('Accuracy/train_losses.npy', np.array(train_losses))
    np.save('Accuracy/train_accs.npy', np.array(train_accs))
    np.save('Accuracy/test_losses.npy', np.array(test_losses))
    np.save('Accuracy/test_accs.npy', np.array(test_accs))
