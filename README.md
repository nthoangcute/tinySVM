# tinySVM
> Very simple support vector machine by Java
## INSTALLING
Download the MNIST or EMNIST dataset from [here](https://drive.google.com/drive/folders/10MfF2F5M40NxEFLSpaHWCMo4y8yEMivI?usp=sharing).

Change your path in **trainFile** and **testFile**. 
```
  public String trainFile = "/home/vietbt/java/mnist_digits_train.txt";
  public String testFile = "/home/vietbt/java/mnist_digits_test.txt";
```
More information about MNIST or EMNIST datasets is in [here](https://www.nist.gov/itl/iad/image-group/emnist-dataset). 
After that, run this code with your Java IDE or by linux command line:
```
  javac tinyCNN.java
  java tinyCNN
```
If you have a java memory heap error, reduce *batchSize* to smaller.
## PERFORMANCE
The update version can run with maximum 10 CPUs and return the best result after serval hours.

### MNIST Digits Dataset

* Best test accuracy: 98.47% after 112 steps

<p align="center"><img src="https://lh3.googleusercontent.com/riywUv3Z_yONW9q3gwCOAh7hs9rqXY55_AI7YyxJ7cyI46G3jx5nPks0tZTOtRT0wYe6fwqQe8s90KJJVZP-HFlPR2XeTw6nHZSbgUZraXklPYzw7R00EZNp2cKJpfKxOwk_T79sWA=w2400" width="600"></p>

## AUTHOR
> **Bui The Viet** - *FPT University* - vietpro213tb@gmail.com
