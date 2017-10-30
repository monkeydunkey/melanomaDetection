# Melanoma Detection

This repo contains my work and experimentation for replicating the work of the [ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection) competition winners RECOD Titans.
Their work can be found [here](https://github.com/learningtitans/isbi2017-part3).

## Transfer Learning
Transfer Learning is the technique used to train a Convolutional Neural Network (CNN) to determine chance of a mole to be 
cancerous based on a dermatological photo. The way transfer learning works is that instead of initializing all of a neural
network's weights to some random value and then learn image feature representation from scratch, you resuse the feature representation
learned by a neural network on another dataset. In most cases what people use are benchmark DNN (Deep Neural Network) which 
have performed very well on the ImageNet dataset which contains a 1000 categories with a thousand image in each of those categories.

Transfer Learning can be divided in the following 4 categories
1. Large dataset, similar to the one the neural network was trained on: As the dataset is large we can fine tune through out the 
whole network
2. Small dataset, similar to the one the neural network was trained on: As the dataset is small fine tuning can overfit the model
in such case the best bet would be train the linear classifier such as SVM on the Bottleneck features produced by the model 
3. Large dataset, different to the one the neural network was trained on: Since the dataset is very large, we may expect that 
   we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with 
   weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire 
   network.
4. Small dataset, different to the one the neural network was trained on: As the dataset is small, we cannot train from scratch
 . What could be done in such scenarios is to use the first few layers of the neural and train a linear classifier on the output
 from these initial layer
 
 The Dataset I am working with falls in the 4th category i.e. it is small and different from the Image net dataset on which most
 Deep Neural Networks are trained
 
 ## Experimentations
 Intially wanted to use xception(can be deployed to mobile devices) or inception as they are more efficient and powerful but 
 finally ended up using VGG as it is simple to understand and also removing layers from it does not require in-depth understanding
 of the network architecture, which is not the case with xception and inception. 
 
 1. Load VGG model to keras, remove the top most fully connected layer and use the rest of the network to generate CNN encoding
 of the images i.e. generate  bottle neck features. Train a Linear SVM classifier on top of this feature set. Achieved AUC 0.56
 
 2. Evaluation Metric setup: As their was class imbalance, we needed to track AUC score rather than accuracy, Used callback for
 this task with. Keras does not provide a way to get the validation data in callbacks so stored an image data generator object
 for the validation folder in the callback class
 
 3. Freeze all but the last conv layer and add a new fully connected layer and fine tune it. AUC achieved 0.63 
 
 ## Tips, Tricks and Gotchas
 1. Overfit the network on a small dataset to ensure that the model is actually working
 2. To ensure that eveything is working as expected, check the initial loss at the start of the first epoch, without any regularization
    term, it should be -ln(random class chance), in my case it should have been -ln(0.5) if the classes were balanced
 3. Either use class weights or upsample the minority class so as to avoid the scenario where the model learns just one of the class
 4. Image ordering, it differs between Theano and TensorFlow
 5. If you don't have access to a GPU locally, create a smaller representative dataset for setting up the code and logic 
    testing (data augmentation pipeline, model compilation and a few epochs) before uploading it to a GPU based server.

 4. use `model.summary()` to get and understanding of the network you are working with
 Following are the things to check if the network does not seem to learn much
 1. Batch size: it should not be too large neither small. If large it loses its ability to generalize, small model may only train
 on a single class of sample in 1 batch which can cause the loss to thrash around
 
 2. Data Preprocessing: Ensure that the images are preprocessed in the same way as the DNN you are using for transfer learning
 
 3. Batch: Ensure all the samples in a batch is not of the same class, this can happen if the data is not shuffled before passing
 to the neural network
 
 4. Try Random input: Do this to ensure that the model is actually learning something and that it is performing better than
 what it would on random data
 
 5. Decay Learning Rate: If the loss is sort of oscillating, then it might mean that the learning rate is high for regions near
  convergence
 
