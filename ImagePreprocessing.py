'''
Generate the various augmented images required for training
'''

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import applications
import pandas as pd
import os
import numpy as np
import json

def checkCreateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


DATA_DIR = 'Data'
TRAIN_DIR = 'train'
VAL_DIR = 'val'
AUG_DATA_DIR = 'AugData'
BOTTLE_NECK_DIR = 'BottleneckData'

trainingData = pd.read_csv(os.path.join(DATA_DIR, 'ISIC-2017_Training_Part3_GroundTruth.csv'))
labels = trainingData.melanoma.loc[:400]
train_y = labels[:300]
val_y = labels[300:]
img_width = 299
img_height = 299
batch_size = 16
nb_train_samples = 300
nb_validation_samples = 100

def instantiateFolders():
    #Checking and creating the required Datasets
    for direc in [DATA_DIR, AUG_DATA_DIR, BOTTLE_NECK_DIR]:
        checkCreateDir(direc)
        for dataset in [TRAIN_DIR, VAL_DIR]:
            checkCreateDir(os.path.join(direc, dataset))
            for classDir in ['0', '1']:
                checkCreateDir(os.path.join(direc, dataset, classDir))

def generateSaveAugmentedImages(datagenerators):
    train_datagen, validation_datagen = datagenerators
    targetVariables = {}
    samples = [nb_train_samples, nb_validation_samples]
    for i, dataset in enumerate([TRAIN_DIR, VAL_DIR]):
        print 'Augmenting Images from dataset', dataset
        targetVariables[dataset] = []
        generator = datagenerators[i].flow_from_directory(
                os.path.join(DATA_DIR, dataset),
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='binary',
                save_to_dir = os.path.join(AUG_DATA_DIR, dataset),
                shuffle=False)
        it = 0
        for inputs, targets in generator:
            print it
            targetVariables[dataset].extend(targets.tolist())
            it += 1
            if it > (2 * samples[i]) // batch_size:
                break  # otherwise the generator would loop indefinitely
    return targetVariables
def getImageAugmenters():
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rotation_range=95,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range = 0.2,
        height_shift_range = 0.2)

    test_datagen = ImageDataGenerator()
    return [train_datagen, test_datagen]


def save_bottlebeck_features(datagenerators):
    train_datagen, validation_datagen = datagenerators
    print 'Starting with creating bottle neck features'
    # build the VGG16 network
    model = applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, TRAIN_DIR),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        train_generator, nb_train_samples // batch_size)
    np.save(open(os.path.join(BOTTLE_NECK_DIR, TRAIN_DIR, 'bottleneck_features_train.npy'), 'w'),
            bottleneck_features_train)
    print('train_generator', train_generator.class_indices, train_generator.classes)
    print 'Train bottle neck saved'
    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(DATA_DIR, VAL_DIR),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        validation_generator, nb_validation_samples // batch_size)
    np.save(open(os.path.join(BOTTLE_NECK_DIR, VAL_DIR, 'bottleneck_features_validation.npy'), 'w'),
            bottleneck_features_validation)
    print(validation_generator.class_indices)
    print 'Test bottle neck saved'




if __name__ == '__main__':
    instantiateFolders()
    imageAug = getImageAugmenters()
    generatedImagesLabels = generateSaveAugmentedImages(imageAug)
    with open('augmentedDataLabels.json', 'w') as f:
        json.dump(generatedImagesLabels, f)
    #save_bottlebeck_features(imageAug)
    #train_top_model()
