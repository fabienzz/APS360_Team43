import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.dataprocessing import loadData,getDatasetSizes


if __name__ == "__main__":

    train_size,val_size,test_size = getDatasetSizes()
    print(train_size,val_size,test_size)
    train_dataset  = loadData('data\\Training_small',300,100,train_size)
    val_dataset  = loadData('data\\Validation_small',300,100,val_size)
    test_dataset  = loadData('data\\Test_small',300,100,test_size)


