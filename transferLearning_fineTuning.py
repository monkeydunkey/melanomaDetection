from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import roc_auc_score
import numpy as np

# path to the model weights files.
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'Data/train_sam'
validation_data_dir = 'Data/val_sam'
nb_train_samples = 24 #1800
nb_validation_samples = 24 #200
epochs = 10
batch_size = 12 #25

class auc_roc_callback(keras.callbacks.Callback):
    def __init__(self, val_data_generator, val_labels):
        self.val_data_generator = val_data_generator
        self.val_labels = val_labels
        self.val_samples = val_labels.shape[0]

    def on_train_begin(self, logs={}):
        self.auc_history = []
        self.loss = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.val_data_generator, self.val_samples)
        self.auc_history.append(roc_auc_score(self.val_labels, y_pred))
        print '\n AUC Score:: ', self.auc_history[-1]
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        return

# build the VGG16 network
input = Input(shape=(3, img_width, img_height),name = 'image_input')
model = applications.VGG16(weights='imagenet', include_top=False, input_tensor = input)
print('Model loaded.', model.output_shape[1:])

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

for layer in model.layers:
    layer.trainable = False

mdl = Model(input= model.input, output= top_model(model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
mdl.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255)
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


#Creating the callbacks
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

val_labels = np.array([0]*20 + [1]*4)#np.array([0]*167 + [1]*33)
checkpointer = ModelCheckpoint(filepath="trainedmodel/weights.hdf5", verbose=1, save_best_only=True)
auc_roc_hist = auc_roc_callback(val_generator, val_labels)

class_weight = {0 : 1.,
    1: 6.}

# fine-tune the model
mdl.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[auc_roc_hist, checkpointer],
    class_weight = class_weight)
print 'AUC Scores are', auc_roc_hist.auc_history
