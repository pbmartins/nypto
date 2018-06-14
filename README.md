## "Offline" model (master branch)

### Default model with 39 classes (window aggregation: 1) - SVM SVC

True positives =  1201
False negatives =  1455
False positives =  704
True negatives =  11469
Precision =  0.6304461942257218
Recall =  0.45218373493975905
Accuracy =  0.8544069053880909

## Default model with 39 classes (window aggregation: 40, threshold: 0.55) - SVM SVC

True positives =  1269
False negatives =  1452
False positives =  636
True negatives =  11472
Precision =  0.6661417322834645
Recall =  0.4663726571113561
Accuracy =  0.8591948209589318

## Live filtering model with window aggregation (live-filtering branch) 

### Binary classification (2 aggregated classes) - SVM Linear SVC

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

### All classes with binary results (window aggregation: 20, threshold: 0.6) - SVM SVC

Confusion Matrix:

```
     0     1     2     3     4     5     6     7     8     9    10    11    12    13 
 0 149.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
 1  11.0 156.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
 2   0.0   5.0 142.0   4.0   0.0   7.0   7.0   3.0   0.0   0.0   0.0   2.0   0.0   2.0 
 3   0.0   1.0   4.0 138.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
 4   0.0   0.0   0.0   0.0   0.0   0.0   6.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
 5   0.0   0.0   0.0   0.0   0.0  61.0   5.0  18.0   0.0   0.0   2.0  88.0   0.0   0.0 
 6   0.0   0.0   0.0   0.0   0.0 128.0  64.0   1.0   0.0   0.0   0.0   7.0   0.0   0.0 
 7   0.0   0.0   0.0   0.0   0.0  13.0  10.0  40.0  30.0   0.0   2.0  85.0   1.0   2.0 
 8   0.0   0.0   0.0   0.0   0.0   0.0   5.0  12.0  41.0   4.0   5.0  54.0  26.0   6.0 
 9   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 950.0   0.0   0.0 
10   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 1978.0   0.0   0.0 
11   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 6150.0   0.0   0.0 
12   0.0   0.0   0.0   0.0   0.0   9.0  96.0  14.0   0.0   0.0   0.0  78.0   0.0   0.0 
13   0.0   0.0   0.0   0.0   0.0   5.0   1.0   1.0   4.0   0.0   1.0 162.0  16.0   1.0 
```

True positives =  888
False negatives =  139
False positives =  123
True negatives =  9663
Precision =  0.8783382789317508
Recall =  0.8646543330087634
Accuracy =  0.9757699065939147
