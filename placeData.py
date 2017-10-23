import pandas as pd
import numpy as np
import shutil
import os

DATA_DIR = 'Data'
TRAIN_DIR = 'train'
VAL_DIR = 'val'
trainingData = pd.read_csv(os.path.join(DATA_DIR, 'ISIC-2017_Training_Part3_GroundTruth.csv'))
x = []
labels = trainingData.melanoma
totalImages = trainingData.image_id.unique().shape[0]
for i, img in enumerate(trainingData.image_id.unique()):
    inPath = os.path.join(DATA_DIR, 'ISIC-2017_Training_Data', img + '.jpg')
    isMel = '1' if labels[i] == 1 else '0'
    outpath = os.path.join(DATA_DIR, TRAIN_DIR if i < totalImages * 0.9 else VAL_DIR, isMel,img + '.jpg')
    shutil.copy(inPath,outpath)
