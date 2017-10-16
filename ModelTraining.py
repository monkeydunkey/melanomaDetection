
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import applications
import pandas as pd
import os
import keras
from sklearn.metrics import roc_auc_score

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




class auc_roc_history(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        print self.model.inputs
        return

    def on_epoch_end(self, epoch, logs={}):
        print self.model.inputs, self.model.targets
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.inputs, verbose = 0)
        auc_score = roc_auc_score(self.model.targets, y_pred)
        self.aucs.append(auc_score)
        logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def train_top_model():
    train_data = np.load(open(os.path.join(BOTTLE_NECK_DIR, TRAIN_DIR, 'bottleneck_features_train.npy')))
    train_labels = np.load(open(os.path.join(BOTTLE_NECK_DIR, TRAIN_DIR, 'bottleneck_labels_train.npy')))

    validation_data = np.load(open(os.path.join(BOTTLE_NECK_DIR, VAL_DIR, 'bottleneck_features_validation.npy')))
    validation_labels = np.load(open(os.path.join(BOTTLE_NECK_DIR, TRAIN_DIR, 'bottleneck_labels_test.npy')))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
