import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch
import torch.optim as optim
from torch.autograd import Variable


"""
Here we load the dataset, add gaussian,poisson,speckle

    'gauss'     Gaussian-distributed additive noise.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.

We define a function that adds each noise when called from main function
Input & Output: np array

"""


def add_noise(img, noise_type="gaussian", mean=0, var=1):
    row, col = 28, 28
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
            acc_num += img1[i][j]**2
            acc_den += (img1[i][j] -  img2[i][j])** 2

    return acc_num / acc_den


def psnr(img1, img2):
    sz = img1.shape
    acc_mse = 0

    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            acc_mse += (img1[i][j] - img2[i][j]) ** 2

    acc_mse /= (sz[0] * sz[1])

    return 10 * np.log(255**2 / acc_mse)

# Here we load the dataset from keras
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
print("No of training datapoints:{}\nNo of Test datapoints:{}".format(len(xtrain), len(xtest)))

"""
From here onwards,we split the 60k training datapoints into 3 sets each given one type of each noise.
We shuffle them for better generalization.
"""
noises = ["gaussian", "speckle"]
mean = 0
var = 500
a = 2
scale = 2
noise_ct = 0
noise_id = 1
traindata = np.zeros((60000, 28, 28))

for idx in tqdm(range(len(xtrain))):
    #traindata[idx] = add_noise(xtrain[idx], noise_type=noises[noise_id], mean=mean, var=var)
    traindata[idx] = add_noise(xtrain[idx], noise_type=noises[noise_id], mean=a, var=scale)

print("\n{} noise addition completed to images".format(noises[noise_id]))

noise_ct = 0
noise_id = 1
testdata = np.zeros((10000, 28, 28))

# for idx in tqdm(range(len(xtest))):
#
#     if noise_ct < (len(xtest) / 2):
#         noise_ct += 1
#         x = add_noise(xtest[idx], noise_type=noises[noise_id])
#         testdata[idx] = x
#
#     else:
#         print("\n{} noise addition completed to images".format(noises[noise_id]))
#         noise_id += 1
#         noise_ct = 0

for idx in tqdm(range(len(xtest))):
    noise_ct += 1
    x = add_noise(xtest[idx], noise_type=noises[noise_id])
    testdata[idx] = x

print("\n{} noise addition completed to images".format(noises[noise_id]))


"""
Here we Try to visualize, each type of noise that was introduced in the images
Along with their original versions

"""


f, axes=plt.subplots(2,2)

#showing images with gaussian noise
axes[0,0].imshow(xtrain[0],cmap="gray")
axes[0,0].set_title("Original Image")
axes[1,0].imshow(traindata[0],cmap='gray')
axes[1,0].set_title("Noised Image")

#showing images with speckle noise
axes[0,1].imshow(xtrain[25000],cmap='gray')
axes[0,1].set_title("Original Image")
axes[1,1].imshow(traindata[25000],cmap="gray")
axes[1,1].set_title("Noised Image")
plt.show()

snr_estim = snr(xtrain[0], traindata[0])
psnr_estim = psnr(xtrain[0], traindata[0])

print("SNR: ", snr_estim)
print("PSNR: ", psnr_estim)

class noisedDataset(Dataset):

    def __init__(self, datasetnoised, datasetclean, labels, transform):
        self.noise = datasetnoised
        self.clean = datasetclean
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.noise)

    def __getitem__(self, idx):
        xNoise = self.noise[idx]
        xClean = self.clean[idx]
        y = self.labels[idx]

        if self.transform != None:
            xNoise = self.transform(xNoise)
            xClean = self.transform(xClean)

        return (xNoise, xClean, y)


tsfms = transforms.Compose([transforms.ToTensor()])

trainset = noisedDataset(traindata, xtrain, ytrain, tsfms)
testset = noisedDataset(testdata, xtest, ytest, tsfms)

# class CustomDataset(Dataset):
#     def __init__(self, clean_data, noisy_data):
#         self.clean = clean_data
#         self.noisy = noisy_data
#
#     def __len__(self):
#         return len(self.clean)
#
#     def __getitem__(self, index):
#         clean_spec = signal.stft(self.clean[index, :], nperseg=254, noverlap=120)[2].reshape((1, 128, 32))
#         noisy_spec = signal.stft(self.noisy[index, :], nperseg=254, noverlap=120)[2].reshape((1, 128, 32))
#
#         clean_tensor = torch.from_numpy(normalisation(clean_spec)).to(torch.float32)
#         noisy_tensor = torch.from_numpy(normalisation(noisy_spec)).to(torch.float32)
#         return clean_tensor, noisy_tensor
#
#
# def normalisation(X):
#     minX = np.min(X)
#     maxX = np.max(X)
#
#     return (X - minX)/(maxX - minX)

"""
Here , we create the trainloaders and testloaders.
Also, we transform the images using standard lib functions
"""

# clean = pd.read_csv('label.csv').to_numpy()
# noisy = pd.read_csv('train.csv').to_numpy()
#
# test_clean = clean[:10]
# test_noisy = noisy[:10]
#
# training_dataset = CustomDataset(clean[10:900], noisy[10:900])
# validation_dataset = CustomDataset(clean[900:], noisy[900:])

batch_size=32

#trainloader = DataLoader(training_dataset, batch_size=batch_size)
#testloader = DataLoader(validation_dataset, batch_size=batch_size)






trainloader=DataLoader(trainset,batch_size=32,shuffle=True)
testloader=DataLoader(testset,batch_size=1,shuffle=True)

"""
Here, we define the autoencoder model.
"""


class denoising_model(nn.Module):
    def __init__(self):
        super(denoising_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)

        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size = 5, padding = "same"),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 256, kernel_size = 5, padding = "same"),
        #     nn.ReLU(True),
        #     nn.Linear(256, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(True)
        #
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(64, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 256),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 32, kernel_size=5, padding="same"),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 1, kernel_size=5, padding="same"),
        #     nn.Sigmoid(),
        # )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


# We check whether cuda is available and choose device accordingly
if torch.cuda.is_available() == True:
    device = "cuda:0"
else:
    device = "cpu"

model = denoising_model().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)

epochs = 100
l = len(trainloader)
losslist = list()
epochloss = 0
running_loss = 0
for epoch in range(epochs):

    print("Entering Epoch: ", epoch)
    for dirty, clean, label in tqdm((trainloader)):
        dirty = dirty.view(dirty.size(0), -1).type(torch.FloatTensor)
        clean = clean.view(clean.size(0), -1).type(torch.FloatTensor)
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

plt.plot(range(len(losslist)), losslist)

"""Here, we try to visualize some of the results.
  We randomly generate 6 numbers in between 1 and 10k , run them through the model,
  and show the results with comparisons

 """

f, axes = plt.subplots(6, 3, figsize=(20, 20))
axes[0, 0].set_title("Original Image")
axes[0, 1].set_title("Dirty Image")
axes[0, 2].set_title("Cleaned Image")

test_imgs = np.random.randint(0, 10000, size=6)
for idx in range((6)):
    dirty = testset[test_imgs[idx]][0]
    clean = testset[test_imgs[idx]][1]
    label = testset[test_imgs[idx]][2]
    dirty = dirty.view(dirty.size(0), -1).type(torch.FloatTensor)
    dirty = dirty.to(device)
    output = model(dirty)

    output = output.view(1, 28, 28)
    output = output.permute(1, 2, 0).squeeze(2)
    output = output.detach().cpu().numpy()

    dirty = dirty.view(1, 28, 28)
    dirty = dirty.permute(1, 2, 0).squeeze(2)
    dirty = dirty.detach().cpu().numpy()

    clean = clean.permute(1, 2, 0).squeeze(2)
    clean = clean.detach().cpu().numpy()

    axes[idx, 0].imshow(clean, cmap="gray")
    axes[idx, 1].imshow(dirty, cmap="gray")
    axes[idx, 2].imshow(output, cmap="gray")

plt.show()

PATH="model"
torch.save(model.state_dict(),PATH)  # We save the model state dict at PATH
