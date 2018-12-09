# DataAnalyticsLab_EE4708_Competition
This repository contains codes used for the [Kaggle competition](https://www.kaggle.com/c/data-analytics-4708-face-verfication/leaderboard)(Team Name : Gesichtsüberprüfung) for the course Data Analytics Lab(EE4708).

# Dependencies
* Python(used v3.6.5)
* sklearn(used v0.19.1)
* cv2(used v3.1.0)
* scipy.ndimage(used v2.0)

# The Challenge
The competition is a Face Verification challenge : verifying if a pair of face images belong to the same person irrespective of lighting conditions, age and other nuisance factors. Solving face verification robustly requires extraction of features from face images which are invariant to several factors such as age, illumination in the image, face profiles(side/frontal), image resolution etc. Rather than designing these features by hand(which is almost impossible), it is better to have a machine "learn" a feature which is invariant to all these(and more) factors. So, the most important part of this competition was the collection of a wide variety of datasets. `yale_faces.ipynb` has more information on the same. 

In particular, this competition focuses on pairs of images of human faces in frontal position. Each image has only one frontal face and the background will vary across the images. The size of the face in each image is not fixed. The faces in the test images can be captured under varying lighting conditions and can also have other distortions such as faces with and without glasses.

# Dataset Collection
Keeping in mind that the test dataset focuses only on _frontal_ faces(with various other distortions), we have used a combination of the following datasets : 
1. [The Yale Face Database](http://vision.ucsd.edu/content/yale-face-database)
2. [The Yale Face Database](http://vision.ucsd.edu/content/extended-yale-face-database-b-b)
3. [The AT&T Database of Faces](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)

Our best result was achieved with a combination of datasets 1 and 3.

# Dataset Preprocessing and Feature Extraction Steps
Though the ML algorithm will learn the final necessary features from the data, the raw images can be preprocessed to remove certain unnecessary information (nuisance parameters) such as noise, lighting conditions etc. and certain face-related features can be extracted before passing to the algorithm.
* Face Cropping from each image using the haar cascade classifier `haarcascade_frontalface_default.xml` and resizing to either a 32*32 or 64*64 image.
* Denoising(using cv2.fastNlMeansDenoising) : Novel fast algorithm based on the similarity of spatially sampled pixels 
* Gaussian Blur : Uses a gaussian function to smoothen the image and further remove noise.
* Histogram Equalisation:  A technique for adjusting image intensities to enhance contrast.
* Histogram of Oriented Gradients (HOG) : Feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image. 
* Local Binary Pattern Histogram(LBPH): Visual descriptor used for classification in computer vision.

The final input feature vectors were obtained by binning the image after dividing it into a specified number of grids or simply by extending all the pixels into a single vector, both followed by application of an element-by-element distance metric between the feature vectors of the two images. The distance metrics experimented with are absolute difference, squared error difference,euclidean distance and cosine similarity measures.

All the above were experimented with, and our best result was obtained with face cropping followed by denoising, gaussian blur, histogram equalization and LBPH. The resulting image was unrolled to make a vector. For each train/test case, the squared difference of the two vectors followed by a cos(for numerical stability) was used as input feature vector.

# Dimensionality Reduction
Both PCA and LDA were tried out. LDA reduces the number of features to at max number_of_classes-1 which in this case is just one. Hence, LDA proved to be unhelpful. PCA was used with varying number of components in the experiments.

# Models Used
Pairs of similar and dissimilar images had to be generated for training.Tried various options like a balanced dataset and a biased one. Dataset having slightly more dissimilar pairs gave better results. 
 
Logistic Regression, MLP, Decision Trees,Random Forests(Best), Ada Boost(Random Forests) - best, Bagging with Decision Trees,Gaussian Process Classifier, RBF-SVM, LDA and QDA were tried out on the data. The best result was given by **AdaBoost with Random Forests** with a private leaderboard score of **0.96497**.



