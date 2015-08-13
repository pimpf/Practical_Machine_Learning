# Prediction_Assignment_Writeup
Milen Angelov  
August 13, 2015  
## Introduction
A group of enthusiasts who take measurements about themselves regularly with devices like Jawbone Up, Nike FuelBand, and Fitbit were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to predict the manner in which they perform barbell lifts - correctly and incorrectly in 5 different ways.

## Loading and preprocessing the data

```r
# Define global option for knitr
knitr::opts_chunk$set(fig.width=8, fig.height=6, fig.path='figure/',
                      echo=TRUE, warning=FALSE, message=FALSE)

# Load libraries
library(caret);
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(dplyr);
```

```
## 
## Attaching package: 'dplyr'
## 
## The following object is masked from 'package:stats':
## 
##     filter
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(randomForest);
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
# Assure reproducability
set.seed(4321);
```

First let's get data we're going to use and load it into the memory

```r
# download data in case it has not done yet, on a first run
fTrainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv";
fTestUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv";

fTrainDest <- "./data/pml-training.csv";
if (!file.exists(fTrainDest)){
    download.file(fTrainUrl, destfile=fTrainDest, method="curl");
}

fTestDest <- "./data/pml-testing.csv"
if (!file.exists(fTestDest)){
    download.file(fTestUrl, destfile=fTestDest, method="curl");
}

# load data
trainData <- read.csv(fTrainDest, na.strings = c("NA", "", "#DIV/0!"));
testData <- read.csv(fTestDest, na.strings = c("NA", "", "#DIV/0!"));
```

First we will partition the train data. We're going to split it into the, training set and the test set and we're going to use about 75% of this  data to train the model and 25% to test.
This will give me a subset of that data that are just for training and a subset of the data
that adjust for testing.

```r
inTrain <- createDataPartition(y = trainData$classe, p = 0.75, list = FALSE);
training <- trainData[inTrain, ];
testing <- trainData[-inTrain, ];
```

Next you can fit a model, so here we're going to use the train command from the caret package, and so trying to predict type. And we use the tilde and the dot to say use all the other variables in this data frame in order to predict the type. And tell also which of the methods
that we'd like to use - here we're going to use GLM.
