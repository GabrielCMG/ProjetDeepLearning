import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm

import torch
from torch import optim
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


def add_noise(img, noise_type="gaussian", mean=0, var=1):
    row, col = 64, 64
    img = img.astype(np.float32)
    sigma = var ** .5

    if noise_type == "gaussian":
        noise = np.random.normal(mean, sigma, img.shape)
        noise = noise.reshape(row, col)
        img = img + noise
        return img

    if noise_type == "speckle":
        noise = np.random.gamma(mean, scale=sigma, size=img.shape)
        img = img * noise
        return img


def snr(img1, img2):
    sz = img1.shape
    acc_num = 0
    acc_den = 0

    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            acc_num += img1[i][j] ** 2
            acc_den += (img1[i][j] - img2[i][j]) ** 2

    return acc_num / acc_den


def psnr(img1, img2):
    sz = img1.shape
    acc_mse = 0

    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            acc_mse += (img1[i][j] - img2[i][j]) ** 2

    acc_mse /= (sz[0] * sz[1])

    return 10 * np.log(255 ** 2 / acc_mse)


def normalisation(X):
    minX = np.min(X)
    maxX = np.max(X)

    return (X - minX) / (maxX - minX)


data = []
imgs_path = "data"

kernel = np.ones((2, 2), np.uint8)

for filename in os.listdir(imgs_path):
    f = os.path.join(imgs_path, filename)
    data.append(cv2.threshold(cv2.dilate(cv2.imread(f, 0), kernel), 20, 255, cv2.THRESH_BINARY)[1])

random.shuffle(data)

xtrain = data[:14000]
xtest = data[14000:]

noises = ["gaussian", "speckle"]
mean = 0
var = 10000.
lb = 1.1
a = lb
scale = 1 / (lb - 1)
noise_ct = 0
noise_id = 0
traindata = np.zeros((14000, 64, 64))

for idx in tqdm(range(len(xtrain))):
    traindata[idx] = add_noise(xtrain[idx], noise_type=noises[noise_id], mean=mean, var=var)

print("\n{} noise addition completed to images".format(noises[noise_id]))

testdata = np.zeros((956, 64, 64))

for idx in tqdm(range(len(xtest))):
    x = add_noise(xtest[idx], noise_type=noises[noise_id], mean=mean, var=var)
    testdata[idx] = x

print("\n{} noise addition completed to images".format(noises[noise_id]))

f, axes = plt.subplots(2, 3)

# showing images with gaussian noise
axes[0, 0].imshow(xtrain[0], cmap="gray")
axes[0, 0].set_title("Original Image")
axes[1, 0].imshow(traindata[0], cmap='gray')
axes[1, 0].set_title("Noised Image")

# showing images with gaussian noise
axes[0, 1].imshow(xtrain[5000], cmap="gray")
axes[0, 1].set_title("Original Image")
axes[1, 1].imshow(traindata[5000], cmap='gray')
axes[1, 1].set_title("Noised Image")

# showing images with speckle noise
axes[0, 2].imshow(xtrain[10000], cmap='gray')
axes[0, 2].set_title("Original Image")
axes[1, 2].imshow(traindata[10000], cmap="gray")
axes[1, 2].set_title("Noised Image")
plt.show()

snr_estim = snr(xtrain[0], traindata[0])
psnr_estim = psnr(xtrain[0], traindata[0])

print("SNR: ", snr_estim)
print("PSNR: ", psnr_estim)


class noisedDataset(Dataset):
    def __init__(self, datasetnoised, datasetclean, transform):
        self.noise = datasetnoised
        self.clean = datasetclean
        self.transform = transform

    def __len__(self):
        return len(self.noise)

    def __getitem__(self, idx):
        xNoise = self.noise[idx]
        xClean = self.clean[idx]

        if self.transform is not None:
            xNoise = self.transform(xNoise)
            xClean = self.transform(xClean)

        return xNoise, xClean


tsfms = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])

trainset = noisedDataset(traindata, xtrain, tsfms)
testset = noisedDataset(testdata, xtest, tsfms)

batch_size = 32

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=True)


class denoising_model(nn.Module):
    def __init__(self):
        super(denoising_model, self).__init__()
        self.encoderDense = nn.Sequential(
            nn.Linear(32 * 32, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)
        )

        self.decoderDense = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 32 * 32),
            nn.Sigmoid(),
        )

        self.encoderConv1D = nn.Sequential(
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

        self.decoderConv1D = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, padding=1, kernel_size=3),
            nn.Sigmoid(),
        )

        self.encoderConv2D = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        self.decoderConv2D = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, padding=1, kernel_size=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoderConv2D(x)
        x = self.decoderConv2D(x)

        return x



model = denoising_model()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

summary(model, (1, 32, 32))

if torch.cuda.is_available() == True:
    device = "cuda:0"
else:
    device = "cpu"

print(device)


model = denoising_model().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)

epochs = 100
l = len(trainloader)
losslist = list()
epochloss = 0
running_loss = 0


image_test = []
for dirty, clean in testloader:
    image_test = dirty

image_test = image_test.view(1, 1, 32, 32).type(torch.FloatTensor)

image_test = image_test.to(device)

for epoch in range(epochs):

    print("Entering Epoch: ", epoch)
    for dirty, clean in tqdm((trainloader)):
        dirty = dirty.view(dirty.size(0), 1, 32, 32).type(torch.FloatTensor)
        clean = clean.view(clean.size(0), 1, 32, 32).type(torch.FloatTensor)
        dirty, clean = dirty.to(device), clean.to(device)

        # -----------------Forward Pass----------------------
        output = model(dirty)
        loss = criterion(output, clean)
        # -----------------Backward Pass---------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epochloss += loss.item()
    # -----------------Log-------------------------------
    losslist.append(running_loss / l)
    running_loss = 0
    print("======> epoch: {}/{}, Loss:{}".format(epoch, epochs, loss.item()))

    image_res = model(image_test)

    image_res = image_res.view(1, 32, 32)
    image_res = image_res.permute(1, 2, 0).squeeze(2)
    image_res = image_res.detach().cpu().numpy()

    plt.imshow(image_res)
    plt.pause(0.01)


plt.plot(range(len(losslist)), losslist)


f, axes = plt.subplots(6, 4, figsize=(20, 20))
axes[0, 0].set_title("Original Image")
axes[0, 1].set_title("Dirty Image")
axes[0, 2].set_title("Cleaned Image")
axes[0, 3].set_title("Latent Space")

test_imgs = np.random.randint(0, 500, size=6)
for idx in range(6):
    dirty = testset[test_imgs[idx]][0]
    clean = testset[test_imgs[idx]][1]
    dirty = dirty.view(dirty.size(0), 1, 32, 32).type(torch.FloatTensor)
    dirty = dirty.to(device)
    output = model(dirty)
    output_encoder = model.encoderConv2D(dirty)

    output = output.view(1, 32, 32)
    output = output.permute(1, 2, 0).squeeze(2)
    output = output.detach().cpu().numpy()

    dirty = dirty.view(1, 32, 32)
    dirty = dirty.permute(1, 2, 0).squeeze(2)
    dirty = dirty.detach().cpu().numpy()

    clean = clean.permute(1, 2, 0).squeeze(2)
    clean = clean.detach().cpu().numpy()

    latent = output_encoder.detach().cpu().numpy()
    latent = latent.reshape(-1,)

    axes[idx, 0].imshow(clean, cmap="gray")
    axes[idx, 1].imshow(dirty, cmap="gray")
    axes[idx, 2].imshow(output, cmap="gray")
    axes[idx, 3].plot(latent)

plt.show()


SNRListe = []
PSNRListe = []

for dirty, clean in tqdm(testloader):
    dirty = dirty.view(1, 32, 32).type(torch.FloatTensor)
    dirty = dirty.to(device)
    output = model(dirty)

    output = output.detach().cpu().numpy()
    output = output.reshape(32,32)

    dirty = dirty.detach().cpu().numpy()
    dirty = dirty.reshape(32,32)

    clean = clean.detach().cpu().numpy()
    clean = clean.reshape(32,32)
    plt.imshow(clean)
    plt.show()
    plt.imshow(dirty)
    plt.show()
    plt.imshow(output)
    plt.show()

    SNRListe.append(snr(dirty, clean))
    PSNRListe.append(psnr(output, clean))
    break

print(f"Moyenne SNR image bruitée : {np.mean(SNRListe)}")
print(f"Moyenne SNR image bruitée : {np.mean(PSNRListe)}")
