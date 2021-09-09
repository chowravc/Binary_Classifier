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

### Import useful scripts
from datasets import *
from model import *
from training_loop import *

print('model.py: imported packages.')



### Get index with largest probability
def maxProbability(tensor):

	## Get indicess of largest probability
	indices = tensor.argmax(dim=2).squeeze()

	## Return the indices
	return indices



### Main functioning of script
def main():

	#### PART 1: Setting up train and test datasets
	print('\nPART 1: Setting up train and test datasets')

	## Setup test train split

	## Decide batch size: This is the number of train image-labels that will be fed at once.
	## Choose the largest one you can without running out of RAM.
	bs = 64

	## Path to dataset
	dataPath = 'data/defects/'

	## Instantiate the train and test datasets
	train_dataset, test_dataset = TrainDataset(dataPath), TestDataset(dataPath)

	## Define the train dataset loader. This will be shuffled.
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle = True)

	## Define the test dataset loader. This will not be shuffled.
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = bs, shuffle = False)

	# We then load this dataset into a dataloader-- this is what will feed the gpu. Gpu's are hungry
	# so we want to feed them as much as possible. the dataloader will spit out a list that holds multiple
	# images and labels, which can be processed in parallel in the gpu. Generally, you want the batch size to
	# be a number big enough to use up all the gpu's memory, so no resource goes to waste.

	# The axes are:
	# batch, rgb, height, width

	## Load next image-label batch
	imgs, labels = next(iter(train_loader))

	## Check if the dimensions are correct
	print('Input image tensor size:', imgs.size())
	print('Input label tensor size:', labels.size())

	print('done.')

	#### PART 2: Setting up model to train
	print('\nPART 2: Setting up model to train')

	## Instantiate model
	model = BinaryClassifier(3, 2).float()

	## Load next image-label batch
	images, labels = next(iter(train_loader))

	## Compute output from untrained model
	output = model(images)

	## Check if the dimensions are correct
	print('Output mask tensor size:', output.size())

	## Get sigmoid of output
	output = torch.sigmoid(output)

	## Get the output maxs
	maxIndices = maxProbability(output)

	print('Displaying test output for untrained model.')
	## Display output
	# print(maxIndices)

	print('done.')

	#### PART 3: Start training
	print('\nPART 3: Start training.')

	## Check if runs directory exists
	if len(glob.glob('runs/')) == 0:
		os.mkdir('runs/')
		os.mkdir('runs/train/')

	## Check if train directory exists
	elif len(glob.glob('runs/train/')) == 0:
		os.mkdir('runs/train/')

	## Number of train experiments
	expNum = len(glob.glob('runs/train/*'))

	## Current train experiment number
	expNum = expNum + 1

	## Experiment directory path
	expPath = 'runs/train/exp' + str(expNum) + '/'

	## Create experiment directory
	os.mkdir(expPath)

	## Choose training device; GPU/CPU
	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
	print(f"Training on device {device}.")

	## Choose a loss function
	loss_fn = nn.BCEWithLogitsLoss()

	## Choose the learning rate
	lr = 1e-6

	## Mount model to device
	model = model.to(device).float()

	## Instantiate optimizer
	optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = .0005, momentum = .9)

	## Create checkpoints directory
	os.mkdir(expPath + 'checkpoints/')

	## Create weights directory
	os.mkdir(expPath + 'weights/')

	## Start training loop
	training_loop(20000, optimizer, model, loss_fn, train_loader, test_loader, device, expPath)
	print('\nFinished training. Saving final model.')

	## Path to final trained model last
	PATH = expPath + 'weights/last.pth'

	## Save the final trained model
	torch.save(model.state_dict(), PATH)



### Run if script is called directly
if __name__ == '__main__':

	## Call main function
	main()

	# tensor = torch.tensor([
	# 	[[1, 0]],
	# 	[[1, 0]],
	# 	[[0, 55]],
	# 	[[0, 1]]
	# 	])
	# print(tensor.shape)
	# print(maxProbability(tensor))