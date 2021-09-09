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



### Define the neural network
class BinaryClassifier(nn.Module):

	## Initialize nn as an object
	def __init__(self, n_input, n_out):
	
		# Initialize module
		super().__init__()

		# Model input dimensions
		self.inDims = n_input

		# Define first linear layer
		self.lin1 = nn.Linear(n_input, 3)
		
		# Define first activation layer
		self.act = nn.Sigmoid()

		# Define second linear layer
		self.lin2 = nn.Linear(3, n_out)

	## Define forward output function
	def forward(self, x):

		# Apply first linear layer
		out = self.lin1(x)

		# Apply first sigmoid layer
		out = self.act(out)

		# Apply second linear layer
		out = self.lin2(x)

		# Return output
		return out



if __name__ == '__main__':
	
	model = BinaryClassifier(3, 2)

	print(model)