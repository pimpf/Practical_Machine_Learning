# Prediction_Assignment_Writeup
Milen Angelov  
August 13, 2015  
## Introduction
A group of enthusiasts who take measurements about themselves regularly with devices like Jawbone Up, Nike Fuel Band, and Fitbit were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to predict the manner in which they perform barbell lifts - correctly and incorrectly in 5 different ways.

## Loading and preprocessing the data

```r
## Define global option for knitr
knitr::opts_chunk$set(fig.width=6, fig.height=4, fig.path='figure/', 
                      echo=TRUE, warning=FALSE, message=FALSE);

## Load libraries
library(caret);
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest);
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(rpart);
library(pls);
```

```
## 
## Attaching package: 'pls'
## 
## The following object is masked from 'package:caret':
## 
##     R2
## 
## The following object is masked from 'package:stats':
## 
##     loadings
```

```r
library(plyr);
library(knitr);
library(gbm);
```

```
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
```

```r
library(kknn);
```

```
## 
## Attaching package: 'kknn'
## 
## The following object is masked from 'package:caret':
## 
##     contr.dummy
```

First let's get data we're going to use and load it into the memory

```r
## download data in case it has not done yet, on a first run
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

## load data
trainData <- read.csv(fTrainDest, na.strings = c("NA", "", "#DIV/0!"));
testData <- read.csv(fTestDest, na.strings = c("NA", "", "#DIV/0!"));

## Do coherent check
all.equal(colnames(testData)[1:length(colnames(testData)) - 1], 
          colnames(trainData)[1:length(colnames(trainData)) - 1]);
```

```
## [1] TRUE
```

## Data Cleaning and PreProcessing

First, we split the data into two groups: a training set and a test set. To do this, the createDataPartition function is used:

```r
inTrain <- createDataPartition(y = trainData$classe,  ## the outcome data are needed
                               p = 0.75,  ## The percentage of data in the training set
                               list = FALSE);  ## The format of the results
training <- trainData[inTrain, ];
testing <- trainData[-inTrain, ];

## Remove non-zero variance
nzvcol <- nearZeroVar(training);
training <- training[, -nzvcol];


## Remove variables with more than 50% missing values
vars <- sapply(training, function(x) {
    sum(!(is.na(x) | x == ""))
})
nulls <- names(vars[vars < 0.5 * length(training$classe)])

## Remove features not related to prediction like id, timestamp and names
firstCols <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
            "cvtd_timestamp", "new_window", "num_window");

rmCols <- c(firstCols, nulls);
training <- training[, !names(training) %in% rmCols];
```


## Tune the model using different algorithms
The following models will be estimated: Random forest, Least squares discriminant analysis (PLSDA), Gradient boosting machine and k-Nearest Neighbors. 


To modify the resampling method, a trainControl function is used. We will use method "k-fold cross validation", which involves splitting the dataset into k-subsets. For each subset is held out while the model is trained on all other subsets. This process is completed until accuracy is determine for each instance in the dataset, and an overall accuracy estimate is provided.


For pre-processing we will use centering and scaling (default).

```r
## Assure repeatability
set.seed(54321);

#tc <- trainControl(method = "repeatedcv", 
#                   repeats = 3,
#                   classProbs = TRUE, 
#                   allowParallel=TRUE);

tc <- trainControl(method = "cv", number = 3); ## if we set it to 10 it computes too long

rfModel <- train(classe ~ ., data = training, method = "rf", trControl = tc);
rfModel$results;
```

```
##   mtry  Accuracy     Kappa   AccuracySD      KappaSD
## 1    2 0.9885853 0.9855583 0.0005414487 0.0006868399
## 2   27 0.9897405 0.9870211 0.0021981296 0.0027802633
## 3   52 0.9829464 0.9784261 0.0050991480 0.0064501908
```

```r
plsModel <- train(classe ~ ., data = training, method = "pls", trControl = tc);
plsModel$results;
```

```
##   ncomp  Accuracy      Kappa  AccuracySD     KappaSD
## 1     1 0.3174344 0.09365841 0.003917254 0.006307107
## 2     2 0.3413507 0.13879548 0.005257655 0.006132755
## 3     3 0.3727409 0.18174637 0.017238375 0.021460529
```

```r
gbmModel <- train(classe ~ ., data = training, method = "gbm", 
                verbose = FALSE, trControl = tc);
gbmModel$results;
```

```
##   shrinkage interaction.depth n.minobsinnode n.trees  Accuracy     Kappa
## 1       0.1                 1             10      50 0.7475213 0.6798976
## 4       0.1                 2             10      50 0.8549401 0.8161996
## 7       0.1                 3             10      50 0.8993066 0.8725430
## 2       0.1                 1             10     100 0.8193372 0.7713802
## 5       0.1                 2             10     100 0.9043347 0.8789367
## 8       0.1                 3             10     100 0.9391901 0.9230547
## 3       0.1                 1             10     150 0.8537171 0.8149136
## 6       0.1                 2             10     150 0.9309690 0.9126634
## 9       0.1                 3             10     150 0.9595732 0.9488611
##    AccuracySD     KappaSD
## 1 0.009336574 0.012302142
## 4 0.006169253 0.007937827
## 7 0.002822507 0.003431466
## 2 0.004135249 0.005540997
## 5 0.004780560 0.006086774
## 8 0.007008319 0.008901986
## 3 0.006567574 0.008505453
## 6 0.004861967 0.006180313
## 9 0.003892000 0.004924253
```

```r
knnModel <- train(classe ~ ., data = training, method = "kknn", trControl = tc);
knnModel$results;
```

```
##   kmax distance  kernel  Accuracy     Kappa  AccuracySD     KappaSD
## 1    5        2 optimal 0.9798207 0.9744728 0.001942886 0.002460139
## 2    7        2 optimal 0.9798207 0.9744728 0.001942886 0.002460139
## 3    9        2 optimal 0.9798207 0.9744728 0.001942886 0.002460139
```


We can see from the results above that Random Forest and Gradient boosting machine models provides results with highest accuracy, even without using ROC for optimization. 

```r
Methods <- c("Random forest", "Least squares discriminant analysis", 
             "Gradient boosting machine", "k-Nearest Neighbors");
Accuracy <- c(max(rfModel$results$Accuracy), 
              max(plsModel$results$Accuracy),
              max(gbmModel$results$Accuracy),
              max(knnModel$results$Accuracy));
perf <- cbind(Methods, Accuracy);
kable(perf);
```



Methods                               Accuracy          
------------------------------------  ------------------
Random forest                         0.989740495094529 
Least squares discriminant analysis   0.372740861530099 
Gradient boosting machine             0.959573213588486 
k-Nearest Neighbors                   0.979820738065327 

To answer the second part of the assignment Random Forest will be used. We will skip testing against test set of training data.

## Testing against test dataset
Apply the algorithm to the 20 test cases in the test data.

```r
answersFromRF <- predict(rfModel, testData);
answersFromRF;
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Saving results

```r
pml_write_files = function(x){
    n = length(x);
    for(i in 1:n){
        filename = paste0("./answers/problem_id_",i,".txt");
        write.table(x[i],
                    file = filename,
                    quote = FALSE,
                    row.names = FALSE,
                    col.names = FALSE);
  }
}

pml_write_files(answersFromRF);
```


## References
1. [How To Estimate Model Accuracy in R Using The Caret Package](http://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/)
2. [A Short Introduction to the caret Package](https://cran.r-project.org/web/packages/caret/vignettes/caret.pdf)
3. [Predictive Modeling with R and the caret Package](https://www.r-project.org/nosvn/conferences/useR-2013/Tutorials/kuhn/user_caret_2up.pdf)
