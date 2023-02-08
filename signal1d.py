import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchsummary import summary
from torch import optim
from scipy import signal


class config:
    batch_size = 10
    learning_rate = 1e-3
    epochs = 50


def normalisation(X):
    minX = np.min(X)
    maxX = np.max(X)

    return (X - minX) / (maxX - minX)


class CustomDataset(Dataset):
    def __init__(self, clean_data, noisy_data):
        self.clean = clean_data
        self.noisy = noisy_data

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean = normalisation(self.clean[index, :])
        dirty = normalisation(self.noisy[index, :])

        clean_tensor = torch.from_numpy(clean)
        noisy_tensor = torch.from_numpy(dirty)

        clean_tensor = clean_tensor.view(1, -1).type(torch.FloatTensor)
        noisy_tensor = noisy_tensor.view(1, -1).type(torch.FloatTensor)

        return clean_tensor, noisy_tensor


clean = pd.read_csv('label3f.csv').to_numpy()
noisy = pd.read_csv('train3f.csv').to_numpy()

test_clean = clean[:10]
test_noisy = noisy[:10]

training_dataset = CustomDataset(clean[10:900], noisy[10:900])
validation_dataset = CustomDataset(clean[900:990], noisy[900:990])
test_dataset = CustomDataset(clean[:10], noisy[:10])

trainloader = DataLoader(training_dataset, batch_size=config.batch_size)
validloader = DataLoader(validation_dataset, batch_size=config.batch_size)
testloader = DataLoader(test_dataset, batch_size=1)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, padding=1, kernel_size=3),
            nn.Sigmoid(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),

        )

        self.decoder2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder2(x)
        x = self.decoder2(x)

        return x


def trainA(dataloader, model, epoch, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for i, (clean, noisy) in enumerate(tqdm(dataloader)):
        clean = clean.to(device)
        noisy = noisy.to(device)

        optimizer.zero_grad()
        pred = model(noisy)
        curr_loss = loss_fn(pred, clean)
        curr_loss.backward()
        optimizer.step()

        # total_loss += curr_loss
        # if (i+1) % 40 == 0:
        #     print('[Epoch number : %d, Mini-batches: %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, total_loss/(i+1)))

    print('[Epoch number : %d] loss: %.3f' % (epoch + 1, total_loss / 89))


def valA(dataloader, model, epoch, loss_fn, device):
    model.eval()
    total_loss = 0.0
    print('-------------------------')
    with torch.no_grad():
        for i, (clean, noisy) in enumerate(tqdm(dataloader)):
            clean = clean.to(device)
            noisy = noisy.to(device)

            output = model(noisy)
            loss = loss_fn(output, clean)
            total_loss += loss

            pred1D = output.cpu().detach().numpy().reshape(10, 4096)
            clean1D = clean.cpu().detach().numpy().reshape(10, 4096)

            if i % 4 == 0:
                plt.figure()
                plt.plot(np.arange(0, 4096, 1), pred1D[1])
                plt.plot(np.arange(0, 4096, 1), clean1D[1])
                plt.xlim([0, 200])
                plt.show()

        print('[Validation] loss: %.3f' %
              (total_loss / 90))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AE().to(device)
summary(model, (1, 4096))

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
loss_fn = torch.nn.functional.mse_loss

image_test = None
image_clean = None
for clean, dirty in testloader:
    image_test = dirty
    image_clean = clean

image_clean = image_clean.detach().cpu().numpy()
image_clean = image_clean.reshape(-1, )

image_test = image_test.to(device)

for epoch in range(config.epochs):
    print("-------------------| EPOCH %d |-------------------" % epoch)
    trainA(trainloader, model, epoch, loss_fn, optimizer, device)
    # valA(validloader, model, epoch, loss_fn, device)
    print("--------------------------------------------------")

    image_res = model(image_test)
    image_res = image_res.detach().cpu().numpy()
    image_res = image_res.reshape(-1, )

    plt.clf()
    plt.plot(image_clean[:100])
    plt.plot(image_res[:100])
    plt.pause(0.01)

for clean, dirty in testloader:
    clean = clean.detach().cpu().numpy()
    clean = clean.reshape(-1, )

    dirty0 = dirty.detach().cpu().numpy()
    dirty0 = dirty0.reshape(-1, )

    dirty = dirty.to(device)
    dirty = model(dirty)
    dirty = dirty.detach().cpu().numpy()
    dirty = dirty.reshape(-1, )

    clean = clean * 2 - 1
    dirty = dirty * 2 - 1
    dirty0 = dirty0 * 2 - 1

    X = np.abs(signal.stft(clean, nperseg=254, noverlap=120)[2])

    Xorig = np.abs(signal.stft(dirty0, nperseg=254, noverlap=120)[2])

    Xres = np.abs(signal.stft(dirty, nperseg=254, noverlap=120)[2])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    cax = ax.matshow(np.transpose(X), interpolation='bilinear', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    fig.colorbar(cax)
    plt.title('Original Spectrogram')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    cax = ax.matshow(np.transpose(Xorig), interpolation='bilinear', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    fig.colorbar(cax)
    plt.title('Noised Spectrogram')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    cax = ax.matshow(np.transpose(Xres), interpolation='bilinear', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    fig.colorbar(cax)
    plt.title('Unnoised Spectrogram')

    plt.figure()
    plt.plot(np.arange(0, 4096, 1), dirty0, label='Unnoised signal')
    plt.plot(np.arange(0, 4096, 1), clean, label='Original signal')
    plt.plot(np.arange(0, 4096, 1), dirty, label='Noised signal')
    plt.legend()
    plt.xlim([0, 300])
    break
