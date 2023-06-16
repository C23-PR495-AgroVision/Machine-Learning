# AgroVision Machine Learning Repository

## Dataset Description
The data that we used is image datasets that consist of 18 different objects, where 9 objects related to plant crop disease and the other 9 about fruit and vegetable ripeness [[1]](https://data.mendeley.com/datasets/6nxnjbn9w6),[[2]](https://data.mendeley.com/datasets/b6fftwbr2v/2),[[3]](https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification),[[4]](https://arxiv.org/abs/1911.10317). Each of the objects has different amount of class.

Here is the table overview about distribution of the data that our team used for this project.

<div align="center">
  
Tabel 1. Dataset Distribution
 
| No | Object Name               | Category  |   Number of Class | Number of Training Set | Number of Test Set |
|:--:|-------------------------|---------|:---------:|---------:|---------:|
| 1. |  Apple (Apel)             | Plant Crop Disease | 4 | 8014 | 1943 |
| 2. |  Bell Pepper (Paprika)    | Plant Crop Disease | 2 | 4033 | 962 |
| 3. |  Cherry (Ceri)    | Plant Crop Disease | 2 | 4205 | 1574 |
| 4. |  Corn (Jagung)    | Plant Crop Disease | 4 | 7665 | 1855 |
| 5. |  Grape (Anggur)    | Plant Crop Disease | 4 | 7335 | 1825 |
| 6. |  Peach (Persik)    | Plant Crop Disease | 2 | 3566 | 891 |
| 7. |  Potato (Kentang)    | Plant Crop Disease | 3 | 5907 | 1442 |
| 8. |  Strawberry (Stroberi)    | Plant Crop Disease | 2 | 3598 | 900 |
| 9. |  Tomato (Tomat)    | Plant Crop Disease | 9 | 17195 | 4197 |
| 10.|  Bell Pepper (Paprika)    | Vegetable Ripeness | 5 | 424 | 163 |
| 11.|  Chile Pepper (Cabai)    | Vegetable Ripeness | 5 | 452 | 166 |
| 12.|  Tomato (Tomat)    | Vegetable Ripeness | 4 | 1021 | 230 |
| 13.|  Apple (Apel)    | Fruit Ripeness | 2 | 1814 | 237 |
| 14.|  Banana (Pisang)    | Fruit Ripeness | 4 | 11793 | 1685 |
| 15.|  Guava (Jambu)    | Fruit Ripeness | 2 | 846 | 145 |
| 16.|  Lime (Jeruk Nipis)    | Fruit Ripeness | 2 | 1016 | 174 |
| 17.|  Orange (Jeruk Nipis)    | Fruit Ripeness | 2 | 1672 | 239 |
| 18.|  Pomegranate (Delima)    | Fruit Ripeness | 2 | 864 | 159 |

</div>
 
## Modeling & Evaluation
For the modeling part, we mainly used four different kind of model architectures. The first one is the self-created architecture where we defined the detail of each models, such as the layers, neurons, etc. The other three is using the transfer learning approach that referred to the official Keras API documentation [[5]](https://keras.io/api/applications/). We used Xception, MobileNetV2, and DenseNet121. The comparison between these three models is as follows.


<div align="center">
  
Tabel 2. Model Comparison of Xception, MobileNetV2, and DenseNet121

| No | Model Name   | Size (MB)  | Parameters (M) | Depth |
|:--:|:------------:|:----------:|:----------:|:-----:|
| 1. |  Xception    | 88         | 22.9      | 81    |
| 2. |  MobileNetV2 | 14         | 3.5       | 105   |
| 3. |  DenseNet121 | 33         | 8.1       | 242   |
  
</div>
  
Here are the detailed metrics that we got after doing model development on each objects using several different approaches mentioned before.

### Plant Crop Disease
#### Apple
<div align="center">
  
Tabel 3. Metrics of Apple Crop Disease Object
  
| No |               Model              | Accuracy  |   Loss    | F1-Score  |
|:--:|:--------------------------------:|:---------:|:---------:|:---------:|
| 1. |  MobileNetV2 (Non-Augmented 2)  | 1 | 0.000879 | 1 |
| 2. |    Xception (Non-Augmented 1)    | 1 | 0.000626 | 1 |
| 3. |    Xception (Non-Augmented 2)    | 1 | 0.000338 | 1 |
| 4. |   DenseNet121   (Augmented 1)   | 1 | 0.000651 | 1 |
| 5. |   DenseNet121   (Augmented 2)   | 1 | 0.000675 | 1 |
| 6. |   MobileNetV2   (Augmented 1)   | 1 | 0.001026 | 1 |
| 7. |   MobileNetV2   (Augmented 2)   | 1 | 0.000297 | 1 |
| 8. |      Xception (Augmented 1)      | 1 | 0.000412 | 1 |
| 9. |      Xception (Augmented 2)      | 1 | 0.002333 | 1 |
| 10. |  DenseNet121 (Non-Augmented 1)  | 0.9994 | 0.002031 | 0.9995 |
| 11. |  DenseNet121 (Non-Augmented 2)  | 0.9994 | 0.002697 | 0.9995 |
| 12. |  MobileNetV2 (Non-Augmented 1)  | 0.9994 | 0.005050 | 0.9995 |
| 13. |   Self-Created (Augmented 1)   | 0.9984 | 0.006339 | 0.9985 |
| 14. |   Self-Created (Augmented 2)   | 0.9938 | 0.022232 | 0.9940 |
| 15. | Self-Created (Non-Augmented 2) | 0.9788 | 0.135590 | 0.9792 |
| 16. | Self-Created (Non-Augmented 1) | 0.9696 | 0.125299 | 0.9702 |
  
</div>

#### Bell Pepper
<div align="center">
  
Tabel 4. Metrics of Bell Pepper Crop Disease Object
  
| No |                Model               | Accuracy |   Loss   |    F1-Score    |
|:--:|:----------------------------------:|:--------:|:--------:|:--------:|
| 1. | MobileNetV2 (Non-Augmented 1)   | 1 | 0.000050 | 1 |
| 2. | MobileNetV2 (Non-Augmented 2)   | 1 | 0.001048 | 1 |
| 3. | DenseNet121   (Augmented 2)       | 1 | 0.002050 | 1 |
| 4. | MobileNetV2   (Augmented 1)       | 1 | 0.000179 | 1 |
| 5. | MobileNetV2   (Augmented 2)       | 1 | 0.000706 | 1 |
| 6. | Xception (Augmented   1)           | 1 | 0.001602 | 1 |
| 7. | Xception (Augmented   2)           | 1 | 0.012214 | 1 |
| 8. | Self-Created   (Augmented 1)     | 1 | 0.002179 | 1 |
| 9. | Self-Created   (Augmented 2)     | 1 | 0.003563 | 1 |
| 10. | DenseNet121 (Non-Augmented 2)   | 0.9989 | 0.003093 | 0.998959 |
| 11. | Xception (Non-Augmented 1)       | 0.9989 | 0.002842 | 0.998959 |
| 12. | Xception (Non-Augmented 2)       | 0.9989 | 0.002909 | 0.998959 |
| 13. | DenseNet121   (Augmented 1)       | 0.9989 | 0.005044 | 0.998959 |
| 14. | DenseNet121 (Non-Augmented 1)   | 0.9979 | 0.008506 | 0.997918 |
| 15. | Self-Created (Non-Augmented 2) | 0.9875 | 0.043474 | 0.987510 |
| 16. | Self-Created (Non-Augmented 1) | 0.9823 | 0.079027 | 0.982315 |
  
</div>

#### Cherry

<div align="center">
  
Tabel 5. Metrics of Cherry Crop Disease Object

| No |                Model               |     Accuracy    |       Loss      |        F1-Score       |
|:--:|:----------------------------------:|:---------------:|:---------------:|:---------------:|
| 1. | DenseNet121 (Non-Augmented 1)   | 1 | 0.000451 | 1 |
| 2. | DenseNet121 (Non-Augmented 2)   | 1 | 0.000026 | 1 |
| 3. | MobileNetV2 (Non-Augmented 1)   | 1 | 0.000031 | 1 |
| 4. | MobileNetV2 (Non-Augmented 2)   | 1 | 0.000277 | 1 |
| 5. | Self-Created (Non-Augmented 2) | 1 | 0.004201 | 1 |
| 6. | Self-Created (Non-Augmented 1) | 1 | 0.000693 | 1 |
| 7. | Xception (Non   Augmented 1)       | 1 | 0.000196 | 1 |
| 8. | Xception (Non   Augmented 2)       | 1 | 0.0000009 | 1 |
| 9. | DenseNet121   (Augmented 1)       | 1 | 0.000431 | 1 |
| 10. | DenseNet121   (Augmented 2)       | 1 | 0.001912 | 1 |
| 11. | MobileNetV2   (Augmented 1)       | 1 | 0.000066 | 1 |
| 12. | MobileNetV2   (Augmented 2)       | 1 | 0.000272 | 1 |
| 13. | Xception (Augmented   1)           | 1 | 0.000049 | 1 |
| 14. | Xception (Augmented   2)           | 1 | 0.000067 | 1 |
| 15. | Self-Created   (Augmented 1)     | 1 | 0.000573 | 1 |
| 16. | Self-Created   (Augmented 2)     | 0.9993 | 0.002806 | 0.9993 |
  
</div>

#### Corn

<div align="center">
  
Tabel 6. Metrics of Corn Crop Disease Object
  
| No |                Model               |  Accuracy |    Loss   |     F1-Score    |
|:--:|:----------------------------------:|:---------:|:---------:|:---------:|
| 1. | Xception (Augmented   2)           | 0.9919 | 0.0412 | 0.9917 |
| 2. | DenseNet121   (Augmented 1)       | 0.9886 | 0.0397 | 0.9885 |
| 3. | DenseNet121 (Non-Augmented 2)   | 0.9876 | 0.0552 | 0.9871 |
| 4. | DenseNet121   (Augmented 2)       | 0.9870 | 0.0433 | 0.9867 |
| 5. | Xception (Augmented   1)           | 0.9870 | 0.0386 | 0.9866 |
| 6. | DenseNet121 (Non-Augmented 1)   | 0.9859 | 0.0488 | 0.9855 |
| 7. | MobileNetV2   (Augmented 2)       | 0.9859 | 0.0404 | 0.9856 |
| 8. | MobileNetV2   (Augmented 1)       | 0.9854 | 0.0573 | 0.9851 |
| 9. | MobileNetV2 (Non-Augmented 1)   | 0.9849 | 0.0717 | 0.9844 |
| 10. | Xception (Non-Augmented 2)       | 0.9849 | 0.0644 | 0.9844 |
| 11. | Xception (Non-Augmented 1)       | 0.9843 | 0.1160 | 0.9838 |
| 12. | MobileNet V2 (Non-Augmented 2)   | 0.9800 | 0.0926 | 0.9794 |
| 13. | Self-Created   (Augmented 1)     | 0.9800 | 0.0571 | 0.9796 |
| 14. | Self-Created   (Augmented 2)     | 0.9741 | 0.0857 | 0.9734 |
| 15. | Self-Created (Non-Augmented 2) | 0.9498 | 0.3311 | 0.9480 |
| 16. | Self-Created (Non-Augmented 1) | 0.9401 | 0.1670 | 0.9380 |
  
 </div>

#### Grape
<div align="center">
  
Tabel 7. Metrics of Grape Crop Disease Object

| No |                Model               |  Accuracy |    Loss   |     F1-Score    |
|:--:|:----------------------------------:|:---------:|:---------:|:---------:|
| 1. | DenseNet 121 (Non-Augmented 2)   | 1 | 0.0412 | 1 |
| 2. | Xception (Non-Augmented 1)       | 1 | 0.0397 | 1 |
| 3. | DenseNet121   (Augmented 2)       | 1 | 0.0552 | 1 |
| 4. | MobileNetV2   (Augmented 1)       | 1 | 0.0433 | 1 |
| 5. | Xception (Augmented   2)           | 1 | 0.0386 | 1 |
| 6. | MobileNetV2 (Non-Augmented 1)   | 0.9994 | 0.0488 | 0.9994 |
| 7. | MobileNetV2   (Augmented 2)       | 0.9994 | 0.0404 | 0.9994 |
| 8. | Xception (Augmented 1)           | 0.9994 | 0.0573 | 0.9994 |
| 9. | DenseNet121 (Non-Augmented 1)   | 0.9989 | 0.0717 | 0.9989 |
| 10. | MobileNetV2 (Non-Augmented 2)   | 0.9989 | 0.0644 | 0.9988 |
| 11. | DenseNet121   (Augmented 1)       | 0.9989 | 0.1160 | 0.9989 |
| 12. | Xception (Non   Augmented 2)       | 0.9983 | 0.0926 | 0.9983 |
| 13. | Self-Created   (Augmented 2)     | 0.9873 | 0.0571 | 0.9875 |
| 14. | Self-Created (Non-Augmented 1) | 0.9857 | 0.0857 | 0.9860 |
| 15. | Self-Created   (Augmented 1)     | 0.9830 | 0.3311 | 0.9833 |
| 16. | Self-Created (Non-Augmented 2) | 0.9819 | 0.1670 | 0.9824 |

</div>

#### Peach
<div align="center">
  
Tabel 8. Metrics of Peach Crop Disease Object

| No |                Model               |  Accuracy |    Loss   |     F1-Score    |
|:--:|:----------------------------------:|:---------:|:---------:|:---------:|
| 1. | DenseNet121 (Non-Augmented 1)   | 1 | 0.00094 | 1 |
| 2. | DenseNet121 (Non-Augmented 2)   | 1 | 0.00120 | 1 |
| 3. | DenseNet121   (Augmented 2)     | 1 | 0.00099 | 1 |
| 4. | MobileNetV2   (Augmented 1)     | 1 | 0.00196 | 1 |
| 5. | MobileNetV2   (Augmented 2)     | 1 | 0.00036 | 1 |
| 6. | Xception (Augmented   1)        | 1 | 0.00229 | 1 |
| 7. | Xception (Augmented   2)        | 1         | 0.00116 | 1 |
| 8. | MobileNetV2 (Non-Augmented 1)   | 0.9994520 | 0.00197 | 0.9988 |
| 9. | MobileNetV2 (Non-Augmented 2)   | 0.9988780 | 0.00277 | 0.9988 |
| 10. | Xception (Non-Augmented 1)     | 0.9988780 | 0.00159 | 0.9988 |
| 11. | Xception (Non-Augmented 2)     | 0.9988780 | 0.00405 | 0.9988 |
| 12. | DenseNet121   (Augmented 1)    | 0.9988780 | 0.00417 | 0.9988 |
| 13. | Self-Created   (Augmented 2)   | 0.9977550 | 0.02150 | 0.9977 |
| 14. | Self-Created   (Augmented 1)   | 0.9943880 | 0.03340 | 0.9943 |
| 15. | Self-Created (Non-Augmented 2) | 0.9831650 | 0.13861 | 0.9831 |
| 16. | Self-Created (Non-Augmented 1) | 0.9809200 | 0.12781 | 0.9809 |
  
 </div>
 
 #### Potato

<div align="center">
  
Tabel 9. Metrics of Potato Crop Disease Object
 
| No |               Model              | Accuracy |   Loss   |    F1-Score    |
|:--:|:--------------------------------:|:--------:|:--------:|:--------:|
| 1. | DenseNet121 (Non-Augmented 1) | 0.9972 | 0.0095 | 0.9972 |
| 2. | DenseNet121 (Non-Augmented 2) | 0.9972 | 0.0063 | 0.9972 |
| 3. | Xception (Non-Augmented 1)     | 0.9965 | 0.0143 | 0.9965 |
| 4. | Xception (Non-Augmented 2)     | 0.9965 | 0.0202 | 0.9965 |
| 5. | DenseNet121   (Augmented 1)     | 0.9965 | 0.0156 | 0.9965 |
| 6. | MobileNetV2   (Augmented 1)     | 0.9965 | 0.0152 | 0.9965 |
| 7. | MobileNetV2   (Augmented 2)     | 0.9965 | 0.0103 | 0.9965 |
| 8. | MobileNetV2 (Non-Augmented 1) | 0.9951 | 0.0162 | 0.9951 |
| 9. | MobileNetV2 (Non-Augmented 2) | 0.9951 | 0.0318 | 0.9951 |
| 10. | DenseNet121   (Augmented 2)     | 0.9951 | 0.0118 | 0.9951 |

 </div>
 
#### Strawberry

<div align="center">
  
Tabel 10. Metrics of Strawberry Crop Disease Object
 
| No |               Model              |     Accuracy    |       Loss      |        F1-Score       |
|:--:|:--------------------------------:|:---------------:|:---------------:|:---------------:|
| 1. | DenseNet121 (Non-Augmented 1) | 1 | 0.00005605 | 1 |
| 2. | DenseNet121 (Non-Augmented 2) | 1 | 0.00000031 | 1 |
| 3. | MobileNetV2 (Non-Augmented 1) | 1 | 0.00000696 | 1 |
| 4. | MobileNetV2 (Non-Augmented 2) | 1 | 0.00000024 | 1 |
| 5. | Xception (Non-Augmented 1)    | 1 | 0.00000055 | 1 |
| 6. | Xception (Non-Augmented 2)    | 1 | 0.00047643 | 1 |
| 7. | DenseNet121   (Augmented 1)   | 1 | 0.00002332 | 1 |
| 8. | DenseNet121   (Augmented 2)   | 1 | 0.00003996 | 1 |
| 9. | MobileNetV2   (Augmented 1)   | 1 | 0.00112955 | 1 |
| 10. | MobileNetV2   (Augmented 2)  | 1 | 0.00059112 | 1 |

</div>

#### Tomato

<div align="center">
  
Tabel 11. Metrics of Tomato Crop Disease Object
  
| No |               Model              | Accuracy |   Loss   |    F1-Score    |
|:--:|:--------------------------------:|:--------:|:--------:|:--------:|
| 1. | DenseNet121 (Non-Augmented 2) | 0.9916 | 0.0332 | 0.9916 |
| 2. | MobileNetV2   (Augmented 1)     | 0.9911 | 0.0327 | 0.9911 |
| 3. | DenseNet121 (Non0Augmented 1) | 0.9909 | 0.0326 | 0.9909  |
| 4. | MobileNetV2   (Augmented 2)     | 0.9909 | 0.0309 | 0.9909 |
| 5. | MobileNetV2 (Non-Augmented 2) | 0.9904 | 0.0417  | 0.9904 |
| 6. | DenseNet121   (Augmented 1)     | 0.9904 | 0.0288 | 0.9904 |
| 7. | DenseNet121   (Augmented 2)     | 0.9899 | 0.0383  | 0.9899 |
| 8. | MobileNetV2 (Non-Augmented 1) | 0.9888 | 0.0343 | 0.9887 |

</div>
 
### Fruit Ripeness
 
#### Apple

<div align="center">
  
Tabel 12. Metrics of Apple Fruit Ripeness Object

| No |    Model     |   Accuracy   |     Loss     |      F1-Score      |
|:--:|:------------:|:------------:|:------------:|:------------:|
| 1. | Xception     | 0.9915 | 0.0292 | 0.9915 |
| 2. | MobileNetV2 | 0.9957 | 0.0181 | 0.9957 |
| 3. | DenseNet121 | 0.9915 | 0.0274 | 0.9915 |
  
</div>

#### Banana

<div align="center">
  
Tabel 13. Metrics of Banana Fruit Ripeness Object

| No |    Model    |   Accuracy   |     Loss     |   F1-Score   |
|:--:|:-----------:|:------------:|:------------:|:------------:|
| 1. | Xception    | 0.9679 | 0.0881 | 0.9678  |
| 2. | DenseNet121 | 0.9727 | 0.0916 | 0.9722 |
  
</div>

#### Guava

<div align="center">
  
Tabel 14. Metrics of Guava Fruit Ripeness Object
 
| No |    Model    |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:-----------:|:----------:|:----------:|:----------:|
| 1. | Xception    | 1 | 0.0098 | 1 |
| 2. | MobileNetV2 | 1 | 0.0038 | 1 |
| 3. | DenseNet121 | 1 | 0.0099 | 1 |
  
 </div>
 
 #### Lime
 
<div align="center">
  
Tabel 15. Metrics of Lime Fruit Ripeness Object
  
| No |    Model    |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:-----------:|:----------:|:----------:|:----------:|
| 1. | Xception    | 0.9482 | 0.1056 | 0.9471 |
| 2. | MobileNetV2 | 0.9827 | 0.0484 | 0.9825 |
| 3. | DenseNet121 | 0.9942 | 0.0399 | 0.9941 |

</div>

#### Orange

<div align="center">
  
Tabel 16. Metrics of Orange Fruit Ripeness Object

| No |    Model    |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:-----------:|:----------:|:----------:|:----------:|
| 1. | Xception    | 0.9707 | 0.1048 | 0.9706 |
| 2. | MobileNetV2 | 0.9958 | 0.0378 | 0.9958 |
| 3. | DenseNet121 | 0.9916 | 0.0445 | 0.9916 |

</div>

#### Pomegranate

<div align="center">
  
Tabel 17. Metrics of Pomegranate Fruit Ripeness Object

| No |    Model    |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:-----------:|:----------:|:----------:|:----------:|
| 1. | Xception    | 0.9937 | 0.0222 | 0.9926 |
| 2. | MobileNetV2 | 1 | 0.0145 | 1 |
| 3. | DenseNet121 | 0.9937 | 0.0290 | 0.9925 |

</div>

### Vegetable Ripeness
#### Bell Pepper

<div align="center">
  
Tabel 18. Metrics of Bell Pepper Vegetable Ripeness Object

| No |            Model           |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:--------------------------:|:----------:|:----------:|:----------:|
| 1. | DenseNet121 (Version   2) | 0.9938 | 0.0415 | 0.9918 |
| 2. | MobileNetV2 (Version   1)  | 0.9783 | 0.0580 | 0.9717 |
| 3. | MobileNetV2 (Version   2)  | 0.9721 | 0.0726 | 0.9636 |
| 4. | DenseNet121 (Version   1) | 0.9690 | 0.0871 | 0.9592 |
| 5. | Xception (Version 2)       | 0.9628 | 0.1439 | 0.9514 |
| 6. | Xception (Version 1)       | 0.9473 | 0.1538 | 0.9321 |

</div>

#### Chile Pepper

<div align="center">
  
Tabel 19. Metrics of Chile Pepper Vegetable Ripeness Object

| No |            Model           |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:--------------------------:|:----------:|:----------:|:----------:|
| 1. | DenseNet121 (Version   1) | 0.9484 | 0.1850 | 0.9417 |
| 2. | DenseNet121 (Version   2) | 0.9484 | 0.1618 | 0.9454 |
| 3. | MobileNetV2 (Version   2)  | 0.9351 | 0.2705 | 0.9251 |
| 4. | MobileNetV2 (Version   1)  | 0.9251 | 0.2585 | 0.9113 |
| 5. | Xception (Version 2)       | 0.9217 | 0.2612 | 0.9124 |
| 6. | Xception (Version 1)       | 0.9134 | 0.2661 | 0.9024 |
  
</div>

#### Tomato

<div align="center">
  
Tabel 20. Metrics of Tomato Vegetable Ripeness Object

| No |            Model           |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:--------------------------:|:----------:|:----------:|:----------:|
| 1. | DenseNet121 (Version   2) | 0.9774 | 0.1355 | 0.9780 |
| 2. | Xception (Version 2)       | 0.9774 | 0.1618 | 0.9775 |
| 3. | MobileNetV2 (Version   2)  | 0.9718 | 0.1880 | 0.9722 |
| 4. | Xception (Version 1)       | 0.9718 | 0.2087 | 0.9710 |
| 5. | DenseNet121 (Version   1) | 0.9690 | 0.1631 | 0.9700 |
| 6. | MobileNetV2 (Version   1)  | 0.9690 | 0.1950 | 0.9706 |
  
</div>

## Notes
The difference between model names that ended with the letter "1" (e.g. "... Non-Augmented 1", "... Augmented 1", and "... Version 1") and the letter "2" (e.g. "... Non-Augmented 2", "... Augmented 2", and "... Version 2") is related to the layer that was used before the model output layer. Model names that ended with the letter "1" use GlobalMaxPooling2D for the last model layer before the output layer, while model names that ended with the letter "2" use GlobalAveragePooling2D.

## References
[1] Suryawanshi, Yogesh; PATIL, Kailas; Chumchu, Prawit (2022), “VegNet: Vegetable Dataset with quality (Unripe, Ripe, Old, Dried and Damaged)”, Mendeley Data, V1, doi: 10.17632/6nxnjbn9w6.

[2] PATIL, Kailas; MESHRAM, Vishal (2021), “FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed quality)”, Mendeley Data, V2, doi: 10.17632/b6fftwbr2v.2

[3] Roboflow Universe Projects, "Banana Ripeness Classification Dataset," Roboflow Universe, Roboflow, Dec. 2022. [Online]. Available: https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification.

[4] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, "PlantDoc: A Dataset for Visual Plant Disease Detection," in Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, Hyderabad, India, 2020, pp. 249-253, doi: 10.1145/3371158.3371196.

[5] K. Team, "Keras documentation: Keras Applications," Keras.io, 2023. [Online]. Available: https://keras.io/api/applications/.
