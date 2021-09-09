### Import useful packages
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

print('model.py: imported packages.')

### Import useful scripts
from datasets import *
from model import *

def main():

	seed = 0
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	#setup test train split

	# Batch size
	bs = 74

	# Path to dataset
	dataPath = 'data/defects/'

	# Call datasets as object
	train_dataset, test_dataset = TrainDataset(dataPath), TestDataset(dataPath)

	# Define train dataset loader
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle = True)

	# Define test dataset loader
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = bs, shuffle = False)

	data, labels = next(iter(train_loader))

	## Select device, CPU/GPU
	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
	print(f"Classifying on device {device}.")

	model = BinaryClassifier(3, 2).to(device)

	model.load_state_dict(torch.load('runs/train/exp3/weights/best_loss.pth'))
	outTensorLast = torch.sigmoid(model.forward(data.float().to(device)))

	model.load_state_dict(torch.load('runs/train/exp3/weights/best_valid.pth'))
	outTensorBest = torch.sigmoid(model.forward(data.float().to(device)))

	def check_accuracy(labels, output, show=True):
		
		# print(labels)
		# print(output)
		
		correct = 0
		values = []
		
		for i in range(labels.shape[0]):
			
			label = labels[i].numpy()
			out = output[i].cpu().detach().numpy()
			
			groundTruth = label[0][0] > label[0][1] # True mean plus, False means minus
			detection = out[0][0] > out[0][1] # True mean plus, False means minus
			
			values.append([groundTruth, detection])
			
			if groundTruth == detection:
				correct = correct + 1
		if show:
			print(str(100*correct/labels.shape[0])[:5] + '%')
		# print(values)
		return correct

	print('\nLast:')
	check_accuracy(labels, outTensorLast)
	print('\nBest:')
	check_accuracy(labels, outTensorBest)

	# res = []

	# for i in range(20000):

	# 	model.load_state_dict(torch.load('runs/train/exp3/checkpoints/epoch_' + str(i+1).zfill(5) + '.pth'))
	# 	outTensor = torch.sigmoid(model.forward(data.float().to(device)))

	# 	if i == 1 or i%100 == 0:
	# 		print(i)
	# 		res.append(check_accuracy(labels, outTensor, show=True))
	# 	else:
	# 		res.append(check_accuracy(labels, outTensor, show=False))

	# res = np.asarray(res)
	# np.savetxt('res.txt', res)
	# print(np.amax(res))

if __name__ == '__main__':
	
	# data = 100*(np.loadtxt('res.txt')/74)
	# print(np.where(data == np.amax(data)))
	# plt.plot(data)
	# plt.show()

	main()