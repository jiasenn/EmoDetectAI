# EmoDetectAI
The project problem that is being addressed is the need for accurate facial and emotion detection using AI technology. Mainly focusing on training models using deep learning models.

Data used is obtained from kaggle.com and is a dataset of 48x48 pixel grayscale images of faces. The dataset consists of 7 different emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

The model is trained using a convolutional neural network (CNN) and is able to achieve an accuracy of 90% on the test set.

## Getting Started
We mainly uses tensorflow and pytorch for the training and testing of our data. We also use the following libraries:
* numpy
* pandas
* matplotlib
* seaborn

## Model Training
Each notebook correspond to a different model.
* `baseline_cnn.ipynb` - Trains a 3 layer CNN model using pytorch
* `CNN.ipynb` - Trains a 7 layer CNN model using pytorch
* `GoogLEnet.ipynb` - Trains a GoogLeNet CNN model using pytorch
* `MobileNet.ipynb` - Trains a Inception CNN model using tensorflow
* `ResNet.ipynb` - Trains a ResNet CNN model using pytorch


## Running the tests
To run the tests, simply run the OpenCV folder. Two different py file correspond to the pytorch model and the tensorflow model. This will open up a window that will detect your face and display the emotion that is being detected.

cv_pth_video.py runs the pytorch model for real time video emotion detection using the best model we have trained, which is the ResNet model. It has an accuracy of 61.23% on the test set.