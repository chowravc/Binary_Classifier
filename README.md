# Binary Classification Neural Network

This Python repository implements a binary classification neural network in Pytorch with an arbitrary number of inputs and no hidden layers.

## Training information

Create a folder `data/` and put your data set at `data/<data-set>/`.
The format of the data set is:
```
<data-set>
|-- test_set/
|-- train_set/
```
The test set and train set must contain input text files containing numpy arrays called `<filename>.txt` and associated labels (output text files) (numpy arrays to txt) `<filename>.txt` of the same size.
 
After creating this dataset, open `train.py`, and put path to dataset `data/<data-set>/` in line 37. Other values such as batch size, learning rate and model can be chosen here.

You can also choose the number of epochs directly in the training loop.

For the model 'BinaryClassifier', a train input size of 3 is expected with a binary output of two probabilities (size 2).

Finally, run `train.py` with:
```
!python train.py
```

Training results will get stored `runs/train/<exp>/` and one weight will be stored every epoch in `checkpoints/`. The last epoch weights and best epoch weights will be stored in `weights`. An in-depth look at the training is stored to `results.txt`.

## Detection information

All inputs to be processed must go to new directory `data/<new-input-directory>/<your-input-data>.txt`.

After putting inputs here, run:
```
!python -W ignore classify.py --w <path-to-trained-weight> --src data/<your-inputs>/ --model <choose-model>
```

Currently, the model choices is only 'BinaryClassifier'. Make sure your weight was trained for the right model.

Example weight path:
```
runs/train/exp1/weights/best.pth
```

Classification results will be stored in `runs/labels/<exp>/` as text files.
