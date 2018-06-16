## "Offline" model (master branch)

Window size = 2 minutes  
Sliding time = 20 seconds

### Default model with 39 classes (window aggregation: 1) - SVM SVC

True positives =  1201  
False negatives =  1455  
False positives =  704  
True negatives =  11469  
Precision =  0.6304461942257218  
Recall =  0.45218373493975905  
Accuracy =  0.8544069053880909  

### Default model with 39 classes (window aggregation: 40, threshold: 0.55) - SVM SVC

True positives =  1269  
False negatives =  1452  
False positives =  636  
True negatives =  11472  
Precision =  0.6661417322834645  
Recall =  0.4663726571113561  
Accuracy =  0.8591948209589318  

## Live filtering model with window aggregation (live-filtering branch) 

### Binary classification (2 aggregated classes) - SVM Linear SVC

Window size = 2 minutes  
Sliding time = 20 seconds

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

### All classes with binary results (window aggregation: 70, threshold: 0.55) - SVM SVC

Window size = 6 minutes  
Sliding time = 20 seconds  

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
