from matplotlib import pyplot as plt
from IPython import display
import time
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from csv import writer
import os

def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


for modelname in ['b6', 'b5', 'b4', 'b3', 'b2', 'b1', 'b0']:
	for tl_scheme in ['ori', 'dark']:
		print('{} {}'.format(tl_scheme, modelname))
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		use_parallel = False

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

		testset = torchvision.datasets.CIFAR10(root='../data', train=False,
		                                       download=True, transform=val_transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
		                                         shuffle=False, num_workers=2)

		classes = ('plane', 'car', 'bird', 'cat',
		           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

		# define CNN
		import torch.nn as nn
		import torch.nn.functional as F
		from efficientnet_pytorch import EfficientNet

		path_dir = os.path.join("save", "{}_{}".format(tl_scheme, modelname))
		PATH = os.path.join(path_dir, "best_model.pth")

		net = EfficientNet.from_pretrained('efficientnet-{}'.format(modelname), num_classes=10)
		net.load_state_dict(torch.load(PATH))
		net.to(device)

		correct = 0
		total = 0
		with torch.no_grad():
			net.eval()
			j=1
			for data in testloader:
				# print(j)
				j+=1
				images, labels = data[0].to(device), data[1].to(device)
				outputs = net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		end_epoch = time.time()
		acc = (100 * correct / total)
		print('{} {} test acc: {} %'.format(tl_scheme, modelname, acc))

		del net