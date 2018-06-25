## "Offline" model (master branch)

### Best model with 31 classes (window size: 6 min, window slide: 20s, window aggregation: 40, threshold: 0.60) - Random forests for global model and SVM SVC for silence avaliation

True positives =  1702  
False negatives =  1121  
False positives =  111  
True negatives =  10422  
Precision =  0.9387755102040817  
Recall =  0.6029047113000354  
Accuracy =  0.9077568134171907  

### Default model with 39 classes (window size: 2 min, window slide: 20s, window aggregation: 1) - SVM SVC

True positives =  1201  
False negatives =  1455  
False positives =  704  
True negatives =  11469  
Precision =  0.6304461942257218  
Recall =  0.45218373493975905  
Accuracy =  0.8544069053880909  

### Default model with 39 classes (window size: 2 min, window slide: 20s, window aggregation: 40, threshold: 0.55) - SVM SVC

True positives =  1269  
False negatives =  1452  
False positives =  636  
True negatives =  11472  
Precision =  0.6661417322834645  
Recall =  0.4663726571113561  
Accuracy =  0.8591948209589318  

### Default model with 39 classes (window size: 6 min, window slide: 20s, window aggregation: 40, threshold: 0.60) - SVM SVC

True positives =  1189
False negatives =  845
False positives =  624
True negatives =  11863
Precision =  0.655819084390513
Recall =  0.5845624385447394
Accuracy =  0.8988361683079678

## Live filtering model with window aggregation (live-filtering branch) 

### Binary classification with 2 aggregated classes (window aggregation: 50, threshold: 0.55) - SVM Linear SVC

Confusion Matrix:

```
    0       1
0 714.0   297.0 
1  98.0  9704.0
```

True positives =  714  
False negatives =  98  
False positives =  297  
True negatives =  9704  
Precision =  0.7062314540059347  
Recall =  0.8793103448275862  
Accuracy =  0.9634698973457875  

### All classes with binary results (window size: 6 min, window slide: 20s, window aggregation: 70, threshold: 0.55) - SVM SVC

Confusion Matrix:

```
        0     1     2     3     4     5     6     7     8     9    10    11    12    13 
  0 140.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  1   0.0 159.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  2   0.0   2.0 106.0  56.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  3   0.0   0.0   0.0 135.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  4   0.0   0.0   0.0  10.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  5   0.0   0.0   0.0  21.0   0.0 145.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  6   0.0   0.0   0.0   0.0   0.0 160.0  32.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  7   0.0   0.0   0.0   0.0   0.0  32.0   0.0  40.0  97.0   0.0   0.0   6.0   0.0   0.0 
  8   0.0   0.0   0.0   0.0   0.0   0.0   0.0  39.0  64.0   0.0   2.0   0.0  39.0   1.0 
  9   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   2.0   0.0   0.0 935.0   5.0   0.0 
 10   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 1970.0   0.0   0.0 
 11   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  11.0   0.0 6129.0   0.0   2.0 
 12   0.0   0.0   0.0   0.0   0.0  31.0  54.0  38.0   0.0   0.0   0.0  59.0   0.0   7.0 
 13   0.0   0.0   0.0   0.0   0.0   0.0  10.0   5.0   0.0   1.0   0.0 167.0   0.0   0.0 
```

True positives =  967  
False negatives =  127  
False positives =  0  
True negatives =  9619  
Precision =  1.0  
Recall =  0.8839122486288848  
Accuracy =  0.9881452440959582  
