### Import useful packages
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims
import pathlib
import torch.optim as optim
from torch.autograd import Variable
import skimage as skm
import glob

print('datasets.py: imported packages.')



### Class defining dataset object
class TrainDataset(Dataset):

	## Initialize object
	def __init__(self, dataPath):
		
		# Load paths to data
		self.data = glob.glob(dataPath + 'train_set/inData/*.txt')
		
		# Replace 'data' with 'labels' to get path to labels
		self.labels = [dataPath.replace('inData', 'labels') for dataPath in self.data]
		
		# Transformation to tensor
		self.to_tensor = transforms.ToTensor()

	## Get length of object
	def __len__(self):
		
		# Return length
		return len(self.data)

	## Function to pull next item from the dataset
	def __getitem__(self, idx):
		
		# If already a tensor, convert to list
		if torch.is_tensor(idx):
			
			# Convert the item to list
			idx.tolist()
		
		# Convert the data to a tensor
		tensor = torch.from_numpy(np.loadtxt(self.data[idx]))
		
		# Add an extra dimension to the tensor
		tensor = tensor.float().unsqueeze(0)
		
		# Get the label as a tensor
		label = torch.from_numpy(np.loadtxt(self.labels[idx]))
		
		# Add an extra dimension to the label
		label = label.float().unsqueeze(0)
		
		# Convert the item to a list of data and label
		sample = [tensor, label]
		
		# Return the item
		return sample

	
	
### Class defining dataset object
class TestDataset(Dataset):

	## Initialize object
	def __init__(self, dataPath):
		
		# Load paths to data
		self.data = glob.glob(dataPath + '/test_set/inData/*.txt')
		
		# Replace 'data' with 'labels' to get path to labels
		self.labels = [dataPath.replace('inData', 'labels') for dataPath in self.data]
		
		# Transformation to tensor
		self.to_tensor = transforms.ToTensor()

	## Get length of object
	def __len__(self):
		
		# Return length
		return len(self.data)

	## Function to pull next item from the dataset
	def __getitem__(self, idx):
		
		# If already a tensor, convert to list
		if torch.is_tensor(idx):
			
			# Convert the item to list
			idx.tolist()
		
		# Convert the data to a tensor
		tensor = torch.from_numpy(np.loadtxt(self.data[idx]))
		
		# Add an extra dimension to the tensor
		tensor = tensor.float().unsqueeze(0)
		
		# Get the label as a tensor
		label = torch.from_numpy(np.loadtxt(self.labels[idx]))
		
		# Add an extra dimension to the label
		label = label.float().unsqueeze(0)
		
		# Convert the item to a list of data and label
		sample = [tensor, label]
		
		# Return the item
		return sample



### Main functioning of script
if __name__ == '__main__':

	#setup test train split

	# Batch size
	bs = 64

	# Path to dataset
	dataPath = 'data/defects/'

	# Call datasets as object
	train_dataset, test_dataset = TrainDataset(dataPath), TestDataset(dataPath)

	# Define train dataset loader
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle = True)

	# Define test dataset loader
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = bs, shuffle = False)

	# we then load this dataset into a dataloader-- this is what will feed the gpu. Gpu's are hungry
	# so we want to feed them as much as possible. the dataloader will spit out a list that holds multiple
	# images and labels, which can be processed in parallel in the gpu. Generally, you want the batch size to
	# be a number big enough to use up all the gpu's memory, so no resource goes to waste.

	# Load the next data and labels
	data, labels = next(iter(train_loader))

	# Display data size
	# print(data)
	print(data.shape)

	# Display labels size
	print(labels.shape)
	# print(labels)