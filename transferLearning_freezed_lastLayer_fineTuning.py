import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'Data/train'
validation_data_dir = 'Data/val'
nb_train_samples = 1800
nb_validation_samples = 200
epochs = 50
batch_size = 25


def save_bottlebeck_features():
    print 'Save bottleneck called'
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, 1800)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)
    print 'train bottleneck feature saved'
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, 200)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)
    print 'val bottleneck feature saved'

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0]*1459 + [1]*341)

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0]*167 + [1]*33)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


#save_bottlebeck_features()
train_top_model()
