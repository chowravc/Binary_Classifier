### Import useful packages
import argparse
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
from PIL import Image
from datetime import datetime

### Import useful scripts
from datasets import *
from model import *

print('model.py: imported packages.')


### Run Binary Classifier model on a folder containing images
def classify_folder(weightPath, imDir, model):

	## Check if runs directory exists
	if len(glob.glob('runs/')) == 0:
		os.mkdir('runs/')
		os.mkdir('runs/classify/')

	## Check if classify directory exists
	elif len(glob.glob('runs/classify/')) == 0:
		os.mkdir('runs/classify/')

	## Number of classify experiments
	expNum = len(glob.glob('runs/classify/*'))

	## Current classify experiment number
	expNum = expNum + 1

	## Experiment directory path
	expPath = 'runs/classify/exp' + str(expNum) + '/'

	## Create experiment directory
	os.mkdir(expPath)

	## Create labels directory
	os.mkdir(expPath + 'labels/')

	## Select device, CPU/GPU
	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
	print(f"Classifying on device {device}.")

	## Mount model to device
	model.to(device)

	## Load trained weights to model
	model.load_state_dict(torch.load(weightPath))

	print('\nLoaded trained BinaryClassifier.\n')

	## Choose data input size
	inSize = model.inDims

	## Create pytorch image loader, will rescale image and crop out the center
	loader = transforms.ToTensor()

	## Read image paths
	dataPaths = glob.glob(imDir + '*')

	## Total number of data arrays
	dataTot = len(dataPaths)

	## For every data path
	for i, dataPath in enumerate(dataPaths):

		## Start time
		startTime = datetime.now()

		## Load data as tensor
		data = torch.from_numpy(np.loadtxt(dataPath)).unsqueeze(0).unsqueeze(0).to(device).float()

		## Run model on data and get output as numpy array
		rawOut = model(data)

		## Run sigmoid on rawOut and get output as numpy array
		rawOut = torch.sigmoid(rawOut)

		## End time
		endTime = datetime.now()

		## Delta time
		deltaTime = endTime - startTime

		## Convert delta time to seconds
		deltaTime = deltaTime.seconds + (1e-3)*deltaTime.microseconds

		## Display message
		message = 'data ' + str(i+1) + '/' + str(dataTot) + ': ' + str(dataPath) + ' ' + str(inSize) + ' input features, Done. (' + str(deltaTime) + 's)'
		print(message)

		## Get numpy array from output tensor
		outRay = rawOut[0][0].to("cpu").detach().numpy()

		## Path to output label
		pathOutLabel = expPath + 'labels/' + dataPath.split('\\')[-1]

		## Save numpy array as label txt
		np.savetxt(pathOutLabel, outRay)



### Main functioning of script
def main(args):

	## Path to trained model
	weightPath = args.w

	## Path to weights
	imDir = args.src

	## Load Binary Classifier Model
	model = BinaryClassifier(3, 2)

	## Run Classification on folder
	classify_folder(weightPath, imDir, model)



### Main functioning of script
if __name__ == '__main__':

	## Call new argument parser
	parser = argparse.ArgumentParser()

	## Add weights argument
	parser.add_argument('--w', action='store', nargs='?', type=str, default='runs/train/exp1/weights/best.pth', help='Path to model trained weights (.pth).')

	## Add image directory argument
	parser.add_argument('--src', action='store', nargs='?', type=str, default='data/testData/', help='Path to directory containing data for classification.')

	## Parse all arguments
	args = parser.parse_args()

	## Call main with arguments
	main(args)