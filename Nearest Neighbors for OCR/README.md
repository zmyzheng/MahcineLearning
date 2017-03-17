# Nearest-Neighbors-for-OCR

1. ocr.mat: OCR image data
  
  The unlabeled training data (i.e., feature vectors) are contained in a matrix called data (one
  point per row), and the corresponding labels are in a vector called labels. The test feature vectors
  and labels are in, respectively, testdata and testlabels.


2. NNC_1.py
  
  Write a function that implements the 1-nearest neighbor classifier with Euclidean distance.
  The function takes as input a matrix of training feature vectors and a vector of the
  corresponding labels, as well as a matrix of test feature vectors. The output is a
  vector of predicted labels for all the test points.
  
  Instead of using the 1-NN code directly with data and labels as the training data, do the
  following. For each value n {1000; 2000; 4000; 8000},
  
    (1) Draw n random points from data, together with their corresponding labels.
    
    (2)Use these n points as the training data and testdata as the test points, and compute the
      test error rate of the 1-NN classifier.
    
    Repeat the (random) process described above ten times, independently. Produce an estimate
    of the learning curve plot using the average of these test error rates (that is, averaging over ten
    repetitions). Add error bars to your plot that extend to one standard deviation above and below
    the means. Ensure the plot axes are properly labeled.
    

3. NNC_1_modified.py

  Prototype selection is a method for speeding-up
  nearest neighbor search that replaces the training data with a smaller subset of prototypes (which
  could be data points themselves). For simplicity, assume that 1-NN is used with Euclidean distance.
  
  This code designs a method for choosing prototypes, where the goal is for the 1-NN classifier based on
  the prototypes to have good test accuracy as follows:
  
  (1)	Put the training data with same label into one group, after this process, the data will be separated into 10 groups.
  
  (2)	In every group, calculate the mean value of each point, this will reflect the average value distribution of each character (0-9)
  
  (3)	If we hope to select m prototypes, we will have m/10 prototypes for every character. We first calculate the Euclidean distance between each training data and the means we calculated in step 2 in each group, then sort the data according to the value of distance. From the sorted data, we choose the m/10 prototypes with the same interval. After this step, the selected samples can cover all the situation in each group. And these samples are the prototypes we choose.

