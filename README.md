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

<p align="center"><img src="https://lh3.googleusercontent.com/qgHD21CZatd_o_qa2hNELTEkcKiZtdBffl8E6gQOGBA1QixOGRMt3SN2VDOEWve60LLFCgG5l_DZp1stPhvGeGDCRMAPM7sGaEcKsK3MNTk_Y1DOkK3KxKIfv3OTwmcnxAv4f1vIvg=w2400" width="600"></p>

* Confusion matrix:
  
| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | **993** | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | 0 | 
| **1** | 2 | **993** | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 
| **2** | 2 | 4 | **989** | 4 | 2 | 0 | 0 | 2 | 0 | 2 | 
| **3** | 0 | 1 | 6 | **984** | 0 | 4 | 1 | 4 | 8 | 1 | 
| **4** | 1 | 2 | 1 | 1 | **984** | 1 | 2 | 3 | 3 | 5 | 
| **5** | 0 | 0 | 1 | 5 | 2 | **991** | 4 | 2 | 4 | 5 | 
| **6** | 0 | 0 | 0 | 0 | 0 | 3 | **992** | 2 | 2 | 2 | 
| **7** | 1 | 0 | 1 | 1 | 2 | 0 | 0 | **983** | 5 | 7 | 
| **8** | 1 | 0 | 1 | 4 | 4 | 0 | 0 | 2 | **972** | 11 | 
| **9** | 0 | 0 | 0 | 1 | 5 | 1 | 0 | 2 | 1 | **966** | 


## AUTHOR
> **Bui The Viet** - *FPT University* - vietpro213tb@gmail.com
