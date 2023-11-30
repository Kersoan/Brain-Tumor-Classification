# Brain_Tumor_Classification_Using_Convolutional_Neural_Network

## Overview <a name="overview"></a>
Brain tumors account for 85% to 90% of all primary central nervous system tumors around the world, with the highest incidence and mortality belonging to high HDI regions. With some image classification techniques, I was able to train a model which could then not only determine the presence of a tumor from Brain MRI Scan but also classify the tumor into one of the following types: Glioma, Meningioma, Pituitary Tumor.

## Feature Extraction:
In brain tumor classification using Convolutional Neural Networks (CNNs), feature extraction plays a crucial role in identifying distinctive patterns within medical images. CNNs are a class of deep learning models designed to automatically learn hierarchical features from input data. In the context of brain tumor classification, the CNN is trained on a dataset of medical images, such as MRI scans. During training, the network learns to extract relevant features, such as tumor shape, texture, and spatial relationships, by applying convolutional filters across the input images. These learned features enable the CNN to discern subtle differences between tumor and non-tumor regions. By leveraging the power of deep learning, CNNs have shown promising results in accurately classifying brain tumors, aiding healthcare professionals in diagnosis and treatment planning.

## Getting Started <a name="getting-started"></a>

### Dependencies <a name="dependencies"></a>
* Python 3.*
* Libraries: NumPy, Pandas, Seaborn, Matplotlib, cv2, Keras, tqdm
* Google Colaboratory

### Installation <a name="installation"></a>

* Datasets: The complete set of files is publicly available and can be downloaded from Kaggle. Alternatively, you can find the folder (titled _Brain-MRI_) in my Github repository [here](https://github.com/nazianafis/Brain-Tumor-Classification/tree/main/Brain-MRI).
* Others: The code can be run in [Google Colab](https://colab.research.google.com/github/nazianafis/Brain-Tumor-Classification/blob/main/Brain_Tumor_Classify.ipynb) as an Interactive Python Notebook (ipynb). No additional installation is required.
    - Colaboratory allows you to use and share Jupyter notebooks with others without having to download, install, or run anything on your own computer (other than a browser).

### Project Motivation <a name="project-motivation"></a>

The Project builds a model that is trained on images of Brain MRI Scans, which it then uses to classify a test image into one of the following four categories : 

* Glioma,
* Meningioma,
* Pituitary Tumor, or
* No Tumor

> **Gliomas:** These are the tumors that occur in the brain and/or spinal cord. Types of glioma include: Astrocytomas, Ependymomas, and Oligodendrogliomas. Gliomas are one of the most common types of primary brain tumors. 
> **Meningiomas:** These are the tumors that arise from the Meninges — the membranes that surround the brain and spinal cord. Most meningiomas grow very slowly, often over many years without causing symptoms. 
> **Pituitary tumors:** These are the tumors that form in the Pituitary — a small gland inside the skull. Most pituitary tumors are often pituitary adenomas, benign growths that do not spread beyond the skull.

![dataset](https://github.com/nazianafis/Brain-Tumor-Classification/blob/main/screenshots/dataset.png)

I cropped and augmented the images before building, compiling, training, and evaluating the model.

![crop](https://github.com/nazianafis/Brain-Tumor-Classification/blob/main/screenshots/crop-img.png)

## Results<a name="results"></a>

In the end, I could validate a test image passed through the model.
