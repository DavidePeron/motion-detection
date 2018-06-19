# Motion detection HDA project

## Description
In this project a dataset of *German Aerospace Center (DLR, Deutsches Zentrum für Luft- und Raumfahrt)* is used,
to detect motion from a sensor through a CNN.

---
## Links
- Reference paper: [http://elib.dlr.de/64996/2/activityRecognition.pdf]
- Dataset: [http://www.dlr.de/kn/desktopdefault.aspx/tabid-8500/14564_read-36508/]

---
## Tutorial
Add the dataset to the folder before run any python script. The file is *ARS_DLR_DataSet_V2.mat**

In the Convolutional Neural Network the labels are coded as follows:

| Activity      |           Label |
| ------------- | :-------------: |
| RUNNING       |               0 |
| WALKING       |               1 |
| JUMPING       |               2 |
| STNDING       |               3 |
| SITTING       |               4 |
| XLYINGX       |               5 |
| FALLING       |               6 |
| TRANSUP       |               7 |
| TRANSDW       |               8 |
| TRNSACC       |               9 |
| TRNSDCC       |              10 |
| TRANSIT       |              11 |

*generate_dataset.py* load the .mat file and save the dataset in two different .h5 files: *train_dataset.h5* contains the training set while *test_dataset.h5* contains the test set.

*utility.py* contains some useful functions like *load_dataset()*
