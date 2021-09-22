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
import datetime
import shutil

### Import useful scripts
from model import *

print('training_loop.py: imported packages.')



### Check accuracy of model on test set
def check_accuracy(labels, output, show=True):

	## Store number of correct classifications
	correct = 0

	## Store ground truth, classification
	values = []
	
	## Go through every classification ground truth
	for i in range(labels.shape[0]):
		
		# Convert the ground truth to numpy array
		label = labels[i].numpy()

		# Convert the model output to numpy array after unmounting from CUDA and detacting gradients
		out = output[i].cpu().detach().numpy()
	
		# Check which result is correct	
		groundTruth = label[0][0] > label[0][1] # True mean plus, False means minus

		# Check which result was output
		detection = out[0][0] > out[0][1] # True mean plus, False means minus
		
		# Add them to the list of values
		values.append([groundTruth, detection])
		
		# If equal, the classification is correct
		if groundTruth == detection:

			# Add one to the number of correct results
			correct = correct + 1

	# Ratio of correct to wrong
	r_correct = correct/labels.shape[0]

	## If the user wanted to show a percent
	if show:

		# Display it
		print(str(100*r_correct)[:5] + '%')

	## Return the number of correct classifications and total number
	return correct, labels.shape[0]



### Function defining training loop
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, device, expPath):

	## Store list of losses
	losses = []

	## Store list of validation results
	validation = []

	## Loop through number of epochs
	for epoch in range(1, n_epochs + 1):  # <2>

		# Open results txt file to store training progress
		results = open(expPath + 'results.txt', 'a')

		# Keep track of training loss
		loss_train = 0.0

		# Go through every image-label 'pair' from the train loader
		for imgs, labels in train_loader:  # <3>

			# Load image tensor to GPU/CPU
			imgs = imgs.to(device=device)

			# Load labels tensor to GPU/CPU
			labels = labels.to(device=device)

			# Compute outputs from the model
			outputs = model(imgs)  # <4>

			# Calculate batch loss
			loss = loss_fn(outputs, labels)  # <5>

			# Zero out optimizer gradient
			optimizer.zero_grad()  # <6>

			# Backpropagate loss
			loss.backward()  # <7>

			# Take optimizer step
			optimizer.step()  # <8>

			# Add batch loss to total training loss
			loss_train += loss.item()  # <9>

		# Store total number
		total_test = 0

		# Store correct number
		correct_test = 0

		# Run model on the test set
		for test_imgs, test_labels in test_loader:

			# Load test image tensor to GPU/CPU
			test_imgs = test_imgs.to(device=device)

			# Compute test outputs from the model
			test_outputs = model(test_imgs)

			# Get the ratio of correct detections
			delta_correct_test, delta_test = check_accuracy(test_labels, test_outputs, show=False)

			# Add the delta correct test to total correct
			correct_test = correct_test + delta_correct_test

			# Add the delta total test to total test
			total_test = total_test + delta_test

		# Calculate ratio of correct to total tested
		r_correct = correct_test/total_test

		# Add the ratio to the validation list
		validation.append(r_correct)

		# String to store progress
		progress = '{} Epoch {}, Training loss {}, std {}, teststd {}, validation {}'.format(
				datetime.datetime.now(), epoch, loss_train/len(train_loader), outputs.std(), labels.float().std(), r_correct)

		# Add loss to list
		losses.append(loss_train/len(train_loader))

		# Write progress to results file
		results.write(progress + '\n')

		# Close results file
		results.close()

		# Check if epoch is either 1 or a multiple of 50
		if epoch == 1 or epoch%50 == 0:

			# Display progress every 30 epochs
			print(progress)

		## Path to model checkpoint
		cpPath = expPath + 'checkpoints/epoch_' + str(epoch).zfill(len(str(n_epochs))) + '.pth'

		## Save the final trained model
		torch.save(model.state_dict(), cpPath)

	## Convert losses list to numpy array
	losses = np.asarray(losses)

	## Lowest loss
	lossLow = np.amin(losses)

	## Epoch number with lowest loss
	minEpoch = np.where(losses == lossLow)[-1][0] + 1

	## Display lowest loss
	print('\nLowest loss: ' + str(lossLow) + ', Epoch ' + str(minEpoch))

	## Path to best checkpoint
	cpPath = expPath + 'checkpoints/epoch_' + str(minEpoch).zfill(len(str(n_epochs))) + '.pth'

	## Copy over best checkpoint
	shutil.copyfile(cpPath, expPath + 'weights/best_loss.pth')

	## Convert validation list to numpy array
	validation = np.asarray(validation)

	## Highest validation
	validHigh = np.amax(validation)

	## Epoch number with highest validation
	maxEpoch = np.where(validation == validHigh)[0][0] + 1

	## Display highest validation
	print('\nHighest validation: ' + str(validHigh) + ', Epoch ' + str(maxEpoch))

	## Path to best checkpoint
	cpPath = expPath + 'checkpoints/epoch_' + str(maxEpoch).zfill(len(str(n_epochs))) + '.pth'

	## Copy over best checkpoint
	shutil.copyfile(cpPath, expPath + 'weights/best_valid.pth')



### Run if script is called directly
if __name__ == '__main__':
	
	## Get device GPU/CPU
	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

	## Display found device
	print(f"Training on device {device}.")

	## Choose a loss function
	loss_fn = nn.BCEWithLogitsLoss()

	## Choose learning rate
	lr = 1e-1

	## Instantiate model
	model = BinaryClassifier()

	## Mount model to device
	model = model.to(device)

	## Instantiate optimizer
	optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = .0005, momentum = .9)