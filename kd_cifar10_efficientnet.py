from matplotlib import pyplot as plt
from IPython import display
import time
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from csv import writer
import os
from PIL import Image
import torch.nn.functional as F

from torchvision.datasets import CIFAR10

def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

modelname = 'b0' #student model, option: b0, b1, b2, b3, b4, b5, b6
teacherModelName = 'b1' # b1, b2, b3, b4, b5, b6, b7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
use_parallel = False

temperature = 5

batch_sizes = {
    'b0': 32,
    'b1': 32,
    'b2': 32,
    'b3': 32,
    'b4': 32,
    'b5': 32,
    'b6': 32,
    'b7': 32
}
batch_size = batch_sizes[modelname]
print(batch_size)

resolutions = {
    'b0': 32,
    'b1': 34,
    'b2': 38,
    'b3': 44,
    'b4': 54,
    'b5': 66,
    'b6': 76,
    'b7': 86
}

np.random.seed(int(modelname[-1]))

# define data loader
from torch.utils.data.dataset import Subset

train_indices = list(range(0, 40000))
val_indices = list(range(40000, 50000))

means = (0.4914, 0.4822, 0.4465)
stds = (0.247, 0.243, 0.261)

train_transform = transforms.Compose(
    [transforms.RandomApply([transforms.RandomCrop(28)]),
     transforms.RandomHorizontalFlip(),
     transforms.Resize(resolutions[modelname]),
     transforms.ToTensor(),
     transforms.Normalize(means, stds)])

val_transform = transforms.Compose(
    [transforms.Resize(resolutions[modelname]),
     transforms.ToTensor(),
     transforms.Normalize(means, stds)])

class darkCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(darkCIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)
        self.darkCSVPath = os.path.join("save", "ori_{}".format(teacherModelName), "soft_labels.csv")
        self.darkKnowledge = torch.tensor(np.genfromtxt(self.darkCSVPath, delimiter=',')).float()
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, darkKnowledge = self.data[index], self.targets[index], self.darkKnowledge[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, darkKnowledge, self.data.shape, self.darkKnowledge


train_dataset = darkCIFAR10(root='../data', train=True,
                                        download=True, transform=train_transform)
trainset = Subset(train_dataset, train_indices)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

val_dataset = darkCIFAR10(root='../data', train=True,
                                        download=True, transform=val_transform)
valset = Subset(val_dataset, val_indices)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

testset = darkCIFAR10(root='../data', train=False,
                                       download=True, transform=val_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# import IPython; IPython.embed()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define CNN
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

net = EfficientNet.from_pretrained('efficientnet-{}'.format(modelname), num_classes=10)
if use_parallel:
    net = nn.DataParallel(net, device_ids=[0,1])
net.to(device)

# define loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
darkCriterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

scheduler = None

path_dir = os.path.join("save", "dark_{}".format(modelname))
PATH = os.path.join(path_dir, "best_model.pth")
PATH_CSV = os.path.join(path_dir, "performance.csv")

if(os.path.isfile(PATH)):
    os.remove(PATH)
if(os.path.isfile(PATH_CSV)):
    os.remove(PATH_CSV)

# train the network
print_steps = 20
best_acc = 0
patience = 10
unimproved_count = 0
for epoch in range(100):  # loop over the dataset multiple times
    start_epoch = time.time()
    running_loss = 0.0
    start_iter = time.time()
    for i, data in enumerate(trainloader, 0):
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, darkKnowledge = data[0].to(device), data[1].to(device), data[2].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        darkLoss = darkCriterion(F.log_softmax(outputs/temperature, dim=1), F.softmax(darkKnowledge/temperature, dim=1)) * (temperature * temperature)
        totalLoss = loss + darkLoss
        totalLoss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += totalLoss.item()
        if i % print_steps == print_steps-1:  
            end_iter = time.time()  
            print('[%d, %5d] loss: %.6f | time: %s' %
                  (epoch + 1, i + 1, running_loss / (i+1), hms_string(end_iter - start_iter)))
            start_iter = time.time()

    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in valloader:
            images, labels, darkKnowledge = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            darkLoss = darkCriterion(F.log_softmax(outputs/temperature, dim=1),
                             F.softmax(darkKnowledge/temperature, dim=1)) * (temperature * temperature)
            totalLoss = loss + darkLoss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_epoch = time.time()
    acc = (100 * correct / total)
    train_loss = running_loss / (i+1)
    val_loss = totalLoss.item()
    epoch_interval = end_epoch - start_epoch
    print('val acc: {} %, train loss {}, val loss: {} | time: {}'.format(
       acc, train_loss, val_loss, hms_string(epoch_interval)))
    append_list_as_row(PATH_CSV, [acc, train_loss, val_loss, epoch_interval])
    if acc>best_acc:
        torch.save(net.state_dict(), PATH)
        best_acc = acc
        unimproved_count = 0
    else:
        unimproved_count += 1
    running_loss = 0.0

    if(unimproved_count>=patience):
        print("no improvement for 10 epochs")
        break

    if scheduler is not None:
        scheduler.step()

print('Finished Training')
del net
