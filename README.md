# Brain_Tumor_Classification_Using_Convolutional_Neural_Network

## OVERVIEW <a name="overview"></a>
Brain tumors account for 85% to 90% of all primary central nervous system tumors around the world, with the highest incidence and mortality belonging to high HDI regions. With some image classification techniques, I was able to train a model which could then not only determine the presence of a tumor from Brain MRI Scan but also classify the tumor into one of the following types: Glioma, Meningioma, Pituitary Tumor.

## Feature Extraction:
In brain tumor classification using Convolutional Neural Networks (CNNs), feature extraction plays a crucial role in identifying distinctive patterns within medical images. CNNs are a class of deep learning models designed to automatically learn hierarchical features from input data. In the context of brain tumor classification, the CNN is trained on a dataset of medical images, such as MRI scans. During training, the network learns to extract relevant features, such as tumor shape, texture, and spatial relationships, by applying convolutional filters across the input images. These learned features enable the CNN to discern subtle differences between tumor and non-tumor regions. By leveraging the power of deep learning, CNNs have shown promising results in accurately classifying brain tumors, aiding healthcare professionals in diagnosis and treatment planning.



## Getting Started <a name="getting-started"></a>

### Dependencies <a name="dependencies"></a>
* Python 3.*
* Libraries: NumPy, Pandas, Seaborn, Matplotlib, cv2, Keras, tqdm
* Google Colaboratory

### Installation <a name="installation"></a>

* Datasets: The complete set of files is publicly available and can be downloaded from Kaggle. Alternatively, you can find the folder (titled _Brain-MRI_) in my Github repository.
* Others: The code can be run in as an Interactive Python Notebook (ipynb). No additional installation is required.
    - Colaboratory allows you to use and share Jupyter notebooks with others without having to download, install, or run anything on your own computer (other than a browser).

## Architecture Diagram:
![1](https://github.com/Kersoan/Brain-Tumor-Classification/assets/94525886/abff3989-088c-495b-b471-f253964fec0f)

### Project Motivation <a name="project-motivation"></a>

The Project builds a model that is trained on images of Brain MRI Scans, which it then uses to classify a test image into one of the following four categories : 

* Glioma,
* Meningioma,
* Pituitary Tumor, or
* No Tumor

> **Gliomas:** These are the tumors that occur in the brain and/or spinal cord. Types of glioma include: Astrocytomas, Ependymomas, and Oligodendrogliomas. Gliomas are one of the most common types of primary brain tumors. 
> **Meningiomas:** These are the tumors that arise from the Meninges — the membranes that surround the brain and spinal cord. Most meningiomas grow very slowly, often over many years without causing symptoms. 
> **Pituitary tumors:** These are the tumors that form in the Pituitary — a small gland inside the skull. Most pituitary tumors are often pituitary adenomas, benign growths that do not spread beyond the skull.


## Program
```
import tensorflow as tf
import os
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir

get_ipython().magic(u'matplotlib inline')

def crop_brain_contour(image, plot=False):

    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()
    return new_image

ex_img = cv2.imread('yes/Y1.jpg')
ex_new_img = crop_brain_contour(ex_img, True)

def load_data(dir_list, image_size):
    """
    Read images, resize and normalize them. 
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """
    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size
    for directory in dir_list:
        for filename in listdir(directory):
            print(filename)
            print(directory)
            # load the image
            image = cv2.imread(directory + '/' + filename)
            # print(image)
            # crop the brain and ignore the unnecessary rest part of the image
            image = crop_brain_contour(image, plot=False)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
    X = np.array(X)
    y = np.array(y)
    # Shuffle the data
    X, y = shuffle(X, y)
    print('Number of examples is:{}'.format(len(X)))
    return X,y
augmented_path = 'augmented data/'
# augmented data (yes and no) contains both the original and the new generated examples
augmented_yes = augmented_path + 'yes'
augmented_no = augmented_path + 'no'
IMG_WIDTH, IMG_HEIGHT = (240, 240)
X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

def plot_sample_images(X, y, n=50):

    for label in [0,1]:
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]

        columns_n = 10
        rows_n = int(n/ columns_n)

        plt.figure(figsize=(20, 10))

        i = 1 # current plot        
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])

            # remove ticks
            plt.tick_params(axis='both', which='both',
                            top=False, bottom=False, left=False, right=False,
                           labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            i += 1

        label_to_str = lambda label: "Yes" if label == 1 else "No"
        plt.suptitle("Brain Tumor: {}".format(label_to_str(label)))
        plt.show()

plot_sample_images(X, y)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of development examples = " + str(X_val.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_val (dev) shape: " + str(X_val.shape))
print ("Y_val (dev) shape: " + str(y_val.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))

number of training examples = 1445
number of development examples = 310
number of test examples = 310
X_train shape: (1445, 240, 240, 3)
Y_train shape: (1445, 1)
X_val (dev) shape: (310, 240, 240, 3)
Y_val (dev) shape: (310, 1)
X_test shape: (310, 240, 240, 3)
Y_test shape: (310, 1)

![training-loss1](https://github.com/Kersoan/Brain-Tumor-Classification/assets/94525886/e2d16f15-66e2-4c1e-993a-d8b8129a975d)
![training-accuracy2](https://github.com/Kersoan/Brain-Tumor-Classification/assets/94525886/e15bb844-1cb2-4aed-9886-470b9e934440)


def build_model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. 
    X_input = Input(input_shape)
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) # 

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool0')(X) # 
    # X=Dropout(0.50)(X)


    X = Conv2D(128, (5, 5), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)  # shape=(?, 238, 238, 32)
    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool1')(X) # shape=(?, 14, 14, 32)
    X=Dropout(0.50)(X)

    # FLATTEN X 
    X = Flatten()(X) # shape=(?, 6272)
    # FULLYCONNECTED
    X = Dense(1, activation='sigmoid', name='fc')(X) # shape=(?, 1)


    model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')

    return model
model.metrics_names
loss, acc = model.evaluate(x=X_test, y=y_test)
print ("Test Loss = {}".format(loss))
print ("Test Accuracy = {}".format(acc))

Test Loss = 0.274844214032
Test Accuracy = 0.916129034181

f1score = compute_f1_score(y_test, y_test_prob)
y_val_prob = model.predict(X_val)
f1score_val = compute_f1_score(y_val, y_val_prob)


def data_percentage(y):

    m=len(y)
    n_positive = np.sum(y)
    n_negative = m - n_positive

    pos_prec = (n_positive* 100.0)/ m
    neg_prec = (n_negative* 100.0)/ m

    print("Number of examples:{}".format(m))
    print("Percentage of positive examples: {}%, number of pos examples: {}".format(pos_prec,n_positive))
    print("Percentage of negative examples: {}%, number of neg examples: {}".format(neg_prec,n_negative))

print("Training Data:")
data_percentage(y_train)
print("Validation Data:")
data_percentage(y_val)
print("Testing Data:")
data_percentage(y_test)
```
### Output:
![out1](https://github.com/Kersoan/Brain-Tumor-Classification/assets/94525886/c6252156-ff8e-4e0a-aae1-408ac5440f15)

![out2](https://github.com/Kersoan/Brain-Tumor-Classification/assets/94525886/e275d717-b95c-423a-a08f-a4ba76fb3569)

![out3](https://github.com/Kersoan/Brain-Tumor-Classification/assets/94525886/2352074b-0948-4f6c-bfb1-08fbe0972adf)

## Results<a name="results"></a>

In the end, I could validate a test image passed through the model.
