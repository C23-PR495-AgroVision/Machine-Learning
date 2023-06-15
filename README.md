# AgroVision Machine Learning Repository

## Dataset Description
The data that we used is image datasets that consist of 18 different objects, where 9 objects related to plant crop disease and the other 9 about fruit and vegetable ripeness [[1]](https://data.mendeley.com/datasets/6nxnjbn9w6),[[2]](https://data.mendeley.com/datasets/b6fftwbr2v/2),[[3]](https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification),[[4]](https://arxiv.org/abs/1911.10317). Each of the objects has different amount of class.

Here is the table overview about distribution of the data that our team used for this project.

<div align="center">
  
Tabel 1. Dataset Distribution
 
| No | Object Name               | Category  |   Number of Class | Number of Training Set | Number of Test Set |
|:--:|:-------------------------:|:---------:|:---------:|:---------:|:---------:|
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
 
## Model Evaluation
For the modeling part, we mainly used four different kind of model architectures. The first one is the self-created architecture where we defined the detail of each models, such as the layers, neurons, etc. The other three is using the transfer learning approach that referred to the official Keras API documentation [[5]](https://keras.io/api/applications/). We used Xception, MobileNetV2, and DenseNet121. The comparison between these three models is as follows.


<div align="center">
  
Tabel 2. Model Comparison of Xception, MobileNetV2, and DenseNet121

| No | Model Name   | Size (MB)  | Parameters | Depth |
|:--:|:------------:|:----------:|:----------:|:-----:|
| 1. |  Xception    | 88         | 22.9M      | 81    |
| 2. |  MobileNetV2 | 14         | 3.5M       | 105   |
| 3. |  DenseNet121 | 33         | 8.1M       | 242   |
  
</div>
  
Here are the detailed metrics that we got after doing model development on each objects using several different approaches mentioned before.

### Plant Crop Disease
#### Apple
<div align="center">
  
Tabel 3. Metrics of Apple Crop Disease Object
  
| No |               Model              | Accuracy  |   Loss    | F1-Score  |
|:--:|:--------------------------------:|:---------:|:---------:|:---------:|
| 1. |  MobileNet V2 (Non Augmented 2)  | 1.0000000 | 0.0008790 | 1.0000000 |
| 2. |    Xception (Non Augmented 1)    | 1.0000000 | 0.0006260 | 1.0000000 |
| 3. |    Xception (Non Augmented 2)    | 1.0000000 | 0.0003380 | 1.0000000 |
| 4. |   DenseNet 121   (Augmented 1)   | 1.0000000 | 0.0006510 | 1.0000000 |
| 5. |   DenseNet 121   (Augmented 2)   | 1.0000000 | 0.0006750 | 1.0000000 |
| 6. |   MobileNet V2   (Augmented 1)   | 1.0000000 | 0.0010260 | 1.0000000 |
| 7. |   MobileNet V2   (Augmented 2)   | 1.0000000 | 0.0002970 | 1.0000000 |
| 8. |      Xception (Augmented 1)      | 1.0000000 | 0.0004120 | 1.0000000 |
| 9. |      Xception (Augmented 2)      | 1.0000000 | 0.0023330 | 1.0000000 |
| 10. |  DenseNet 121 (Non Augmented 1)  | 0.9994850 | 0.0020310 | 0.9995030 |
| 11. |  DenseNet 121 (Non Augmented 2)  | 0.9994850 | 0.0026970 | 0.9995030 |
| 12. |  MobileNet V2 (Non Augmented 1)  | 0.9994850 | 0.0050500 | 0.9995030 |
| 13. |   Own Arch (DIY) (Augmented 1)   | 0.9984560 | 0.0063390 | 0.9985050 |
| 14. |   Own Arch (DIY) (Augmented 2)   | 0.9938240 | 0.0222320 | 0.9940360 |
| 15. | Own Arch (DIY) (Non Augmented 2) | 0.9788990 | 0.1355900 | 0.9792920 |
| 16. | Own Arch (DIY) (Non Augmented 1) | 0.9696350 | 0.1252990 | 0.9702690 |
  
</div>

#### Bell Pepper
<div align="center">
  
Tabel 4. Metrics of Bell Pepper Crop Disease Object
  
| No |                Model               | Accuracy |   Loss   |    F1-Score    |
|:--:|:----------------------------------:|:--------:|:--------:|:--------:|
| 1. | MobileNet V2 (Non   Augmented 1)   | 1.000000 | 0.000050 | 1.000000 |
| 2. | MobileNet V2 (Non   Augmented 2)   | 1.000000 | 0.001048 | 1.000000 |
| 3. | DenseNet 121   (Augmented 2)       | 1.000000 | 0.002050 | 1.000000 |
| 4. | MobileNet V2   (Augmented 1)       | 1.000000 | 0.000179 | 1.000000 |
| 5. | MobileNet V2   (Augmented 2)       | 1.000000 | 0.000706 | 1.000000 |
| 6. | Xception (Augmented   1)           | 1.000000 | 0.001602 | 1.000000 |
| 7. | Xception (Augmented   2)           | 1.000000 | 0.012214 | 1.000000 |
| 8. | Own Arch (DIY)   (Augmented 1)     | 1.000000 | 0.002179 | 1.000000 |
| 9. | Own Arch (DIY)   (Augmented 2)     | 1.000000 | 0.003563 | 1.000000 |
| 10. | DenseNet 121 (Non   Augmented 2)   | 0.998960 | 0.003093 | 0.998959 |
| 11. | Xception (Non   Augmented 1)       | 0.998960 | 0.002842 | 0.998959 |
| 12. | Xception (Non   Augmented 2)       | 0.998960 | 0.002909 | 0.998959 |
| 13. | DenseNet 121   (Augmented 1)       | 0.998960 | 0.005044 | 0.998959 |
| 14. | DenseNet 121 (Non   Augmented 1)   | 0.997921 | 0.008506 | 0.997918 |
| 15. | Own Arch (DIY) (Non   Augmented 2) | 0.987526 | 0.043474 | 0.987510 |
| 16. | Own Arch (DIY) (Non   Augmented 1) | 0.982328 | 0.079027 | 0.982315 |
  
</div>

#### Cherry

<div align="center">
  
Tabel 5. Metrics of Cherry Crop Disease Object

| No |                Model               |     Accuracy    |       Loss      |        F1-Score       |
|:--:|:----------------------------------:|:---------------:|:---------------:|:---------------:|
| 1. | DenseNet 121 (Non   Augmented 1)   | 1.0000000000000 | 0.0004516003000 | 1.0000000000000 |
| 2. | DenseNet 121 (Non   Augmented 2)   | 1.0000000000000 | 0.0000262438900 | 1.0000000000000 |
| 3. | MobileNet V2 (Non   Augmented 1)   | 1.0000000000000 | 0.0000312719200 | 1.0000000000000 |
| 4. | MobileNet V2 (Non   Augmented 2)   | 1.0000000000000 | 0.0002778946000 | 1.0000000000000 |
| 5. | Own Arch (DIY) (Non   Augmented 2) | 1.0000000000000 | 0.0042018430000 | 1.0000000000000 |
| 6. | Own Arch (DIY) (Non   Augmented 1) | 1.0000000000000 | 0.0006931162000 | 1.0000000000000 |
| 7. | Xception (Non   Augmented 1)       | 1.0000000000000 | 0.0001961001000 | 1.0000000000000 |
| 8. | Xception (Non   Augmented 2)       | 1.0000000000000 | 0.0000009205923 | 1.0000000000000 |
| 9. | DenseNet 121   (Augmented 1)       | 1.0000000000000 | 0.0004314446000 | 1.0000000000000 |
| 10. | DenseNet 121   (Augmented 2)       | 1.0000000000000 | 0.0019120020000 | 1.0000000000000 |
| 11. | MobileNet V2   (Augmented 1)       | 1.0000000000000 | 0.0000662798000 | 1.0000000000000 |
| 12. | MobileNet V2   (Augmented 2)       | 1.0000000000000 | 0.0002720430000 | 1.0000000000000 |
| 13. | Xception (Augmented   1)           | 1.0000000000000 | 0.0000491197600 | 1.0000000000000 |
| 14. | Xception (Augmented   2)           | 1.0000000000000 | 0.0000672790100 | 1.0000000000000 |
| 15. | Own Arch (DIY)   (Augmented 1)     | 1.0000000000000 | 0.0005738626000 | 1.0000000000000 |
| 16. | Own Arch (DIY)   (Augmented 2)     | 0.9993650000000 | 0.0028064270000 | 0.9993650000000 |
  
</div>

#### Corn

<div align="center">
  
Tabel 6. Metrics of Corn Crop Disease Object
  
| No |                Model               |  Accuracy |    Loss   |     F1-Score    |
|:--:|:----------------------------------:|:---------:|:---------:|:---------:|
| 1. | Xception (Augmented   2)           | 0.9919140 | 0.0412580 | 0.9917700 |
| 2. | DenseNet 121   (Augmented 1)       | 0.9886790 | 0.0397580 | 0.9885180 |
| 3. | DenseNet 121 (Non   Augmented 2)   | 0.9876010 | 0.0552500 | 0.9871870 |
| 4. | DenseNet 121   (Augmented 2)       | 0.9870620 | 0.0433840 | 0.9867490 |
| 5. | Xception (Augmented   1)           | 0.9870620 | 0.0386310 | 0.9866980 |
| 6. | DenseNet 121 (Non   Augmented 1)   | 0.9859840 | 0.0488820 | 0.9855920 |
| 7. | MobileNet V2   (Augmented 2)       | 0.9859840 | 0.0404540 | 0.9856440 |
| 8. | MobileNet V2   (Augmented 1)       | 0.9854450 | 0.0573290 | 0.9851200 |
| 9. | MobileNet V2 (Non   Augmented 1)   | 0.9849060 | 0.0717780 | 0.9844630 |
| 10. | Xception (Non   Augmented 2)       | 0.9849060 | 0.0644340 | 0.9844160 |
| 11. | Xception (Non   Augmented 1)       | 0.9843670 | 0.1160910 | 0.9838680 |
| 12. | MobileNet V2 (Non   Augmented 2)   | 0.9800540 | 0.0926850 | 0.9794230 |
| 13. | Own Arch (DIY)   (Augmented 1)     | 0.9800540 | 0.0571120 | 0.9796400 |
| 14. | Own Arch (DIY)   (Augmented 2)     | 0.9741240 | 0.0857090 | 0.9734730 |
| 15. | Own Arch (DIY) (Non   Augmented 2) | 0.9498650 | 0.3311480 | 0.9480140 |
| 16. | Own Arch (DIY) (Non   Augmented 1) | 0.9401620 | 0.1670510 | 0.9380910 |
  
 </div>

#### Grape
<div align="center">
  
Tabel 7. Metrics of Grape Crop Disease Object

| No |                Model               |  Accuracy |    Loss   |     F1-Score    |
|:--:|:----------------------------------:|:---------:|:---------:|:---------:|
| 1. | DenseNet 121 (Non   Augmented 2)   | 1.0000000 | 0.0412580 | 1.0000000 |
| 2. | Xception (Non   Augmented 1)       | 1.0000000 | 0.0397580 | 1.0000000 |
| 3. | DenseNet 121   (Augmented 2)       | 1.0000000 | 0.0552500 | 1.0000000 |
| 4. | MobileNet V2   (Augmented 1)       | 1.0000000 | 0.0433840 | 1.0000000 |
| 5. | Xception (Augmented   2)           | 1.0000000 | 0.0386310 | 1.0000000 |
| 6. | MobileNet V2 (Non   Augmented 1)   | 0.9994520 | 0.0488820 | 0.9994520 |
| 7. | MobileNet V2   (Augmented 2)       | 0.9994520 | 0.0404540 | 0.9994220 |
| 8. | Xception (Augmented   1)           | 0.9994520 | 0.0573290 | 0.9994220 |
| 9. | DenseNet 121 (Non   Augmented 1)   | 0.9989040 | 0.0717780 | 0.9989310 |
| 10. | MobileNet V2 (Non   Augmented 2)   | 0.9989040 | 0.0644340 | 0.9988750 |
| 11. | DenseNet 121   (Augmented 1)       | 0.9989040 | 0.1160910 | 0.9989310 |
| 12. | Xception (Non   Augmented 2)       | 0.9983560 | 0.0926850 | 0.9983800 |
| 13. | Own Arch (DIY)   (Augmented 2)     | 0.9873970 | 0.0571120 | 0.9875290 |
| 14. | Own Arch (DIY) (Non   Augmented 1) | 0.9857530 | 0.0857090 | 0.9860960 |
| 15. | Own Arch (DIY)   (Augmented 1)     | 0.9830140 | 0.3311480 | 0.9833650 |
| 16. | Own Arch (DIY) (Non   Augmented 2) | 0.9819180 | 0.1670510 | 0.9824680 |

</div>

#### Peach
<div align="center">
  
Tabel 8. Metrics of Peach Crop Disease Object

| No |                Model               |  Accuracy |    Loss   |     F1-Score    |
|:--:|:----------------------------------:|:---------:|:---------:|:---------:|
| 1. | DenseNet 121 (Non   Augmented 1)   | 1.0000000 | 0.0009420 | 1.0000000 |
| 2. | DenseNet 121 (Non   Augmented 2)   | 1.0000000 | 0.0012010 | 1.0000000 |
| 3. | DenseNet 121   (Augmented 2)       | 1.0000000 | 0.0009900 | 1.0000000 |
| 4. | MobileNet V2   (Augmented 1)       | 1.0000000 | 0.0019690 | 1.0000000 |
| 5. | MobileNet V2   (Augmented 2)       | 1.0000000 | 0.0003690 | 1.0000000 |
| 6. | Xception (Augmented   1)           | 1.0000000 | 0.0022940 | 1.0000000 |
| 7. | Xception (Augmented   2)           | 1.0000000 | 0.0011670 | 1.0000000 |
| 8. | MobileNet V2 (Non   Augmented 1)   | 0.9994520 | 0.0019780 | 0.9988770 |
| 9. | MobileNet V2 (Non   Augmented 2)   | 0.9988780 | 0.0027710 | 0.9988770 |
| 10. | Xception (Non   Augmented 1)       | 0.9988780 | 0.0015950 | 0.9988770 |
| 11. | Xception (Non   Augmented 2)       | 0.9988780 | 0.0040590 | 0.9988770 |
| 12. | DenseNet 121   (Augmented 1)       | 0.9988780 | 0.0041770 | 0.9988770 |
| 13. | Own Arch (DIY)   (Augmented 2)     | 0.9977550 | 0.0215090 | 0.9977540 |
| 14. | Own Arch (DIY)   (Augmented 1)     | 0.9943880 | 0.0334000 | 0.9943850 |
| 15. | Own Arch (DIY) (Non   Augmented 2) | 0.9831650 | 0.1386190 | 0.9831530 |
| 16. | Own Arch (DIY) (Non   Augmented 1) | 0.9809200 | 0.1278190 | 0.9809090 |
  
 </div>
 
 #### Potato

<div align="center">
  
Tabel 9. Metrics of Potato Crop Disease Object
 
| No |               Model              | Accuracy |   Loss   |    F1-Score    |
|:--:|:--------------------------------:|:--------:|:--------:|:--------:|
| 1. | DenseNet 121 (Non   Augmented 1) | 0.997226 | 0.009556 | 0.997226 |
| 2. | DenseNet 121 (Non   Augmented 2) | 0.997226 | 0.006319 | 0.997226 |
| 3. | Xception (Non   Augmented 1)     | 0.996533 | 0.014351 | 0.996533 |
| 4. | Xception (Non   Augmented 2)     | 0.996533 | 0.020215 | 0.996533 |
| 5. | DenseNet 121   (Augmented 1)     | 0.996533 | 0.015688 | 0.996533 |
| 6. | MobileNet V2   (Augmented 1)     | 0.996533 | 0.015289 | 0.996533 |
| 7. | MobileNet V2   (Augmented 2)     | 0.996533 | 0.010301 | 0.996533 |
| 8. | MobileNet V2 (Non   Augmented 1) | 0.995146 | 0.016221 | 0.995146 |
| 9. | MobileNet V2 (Non   Augmented 2) | 0.995146 | 0.031861 | 0.995146 |
| 10. | DenseNet 121   (Augmented 2)     | 0.995146 | 0.011837 | 0.995148 |

 </div>
 
#### Strawberry

<div align="center">
  
Tabel 10. Metrics of Strawberry Crop Disease Object
 
| No |               Model              |     Accuracy    |       Loss      |        F1-Score       |
|:--:|:--------------------------------:|:---------------:|:---------------:|:---------------:|
| 1. | DenseNet 121 (Non   Augmented 1) | 1.0000000000000 | 0.0000560572800 | 1.0000000000000 |
| 2. | DenseNet 121 (Non   Augmented 2) | 1.0000000000000 | 0.0000003170912 | 1.0000000000000 |
| 3. | MobileNet V2 (Non   Augmented 1) | 1.0000000000000 | 0.0000069665970 | 1.0000000000000 |
| 4. | MobileNet V2 (Non   Augmented 2) | 1.0000000000000 | 0.0000002448858 | 1.0000000000000 |
| 5. | Xception (Non   Augmented 1)     | 1.0000000000000 | 0.0000005572275 | 1.0000000000000 |
| 6. | Xception (Non   Augmented 2)     | 1.0000000000000 | 0.0004764399000 | 1.0000000000000 |
| 7. | DenseNet 121   (Augmented 1)     | 1.0000000000000 | 0.0000233251400 | 1.0000000000000 |
| 8. | DenseNet 121   (Augmented 2)     | 1.0000000000000 | 0.0000399658700 | 1.0000000000000 |
| 9. | MobileNet V2   (Augmented 1)     | 1.0000000000000 | 0.0011295590000 | 1.0000000000000 |
| 10. | MobileNet V2   (Augmented 2)     | 1.0000000000000 | 0.0005911288000 | 1.0000000000000 |

</div>

#### Tomato

<div align="center">
  
Tabel 11. Metrics of Tomato Crop Disease Object
  
| No |               Model              | Accuracy |   Loss   |    F1-Score    |
|:--:|:--------------------------------:|:--------:|:--------:|:--------:|
| 1. | DenseNet 121 (Non   Augmented 2) | 0.991661 | 0.033225 | 0.991659 |
| 2. | MobileNet V2   (Augmented 1)     | 0.991184 | 0.032725 | 0.991185 |
| 3. | DenseNet 121 (Non   Augmented 1) | 0.990946 | 0.032636 | 0.99094  |
| 4. | MobileNet V2   (Augmented 2)     | 0.990946 | 0.030958 | 0.990949 |
| 5. | MobileNet V2 (Non   Augmented 2) | 0.990469 | 0.04176  | 0.990468 |
| 6. | DenseNet 121   (Augmented 1)     | 0.990469 | 0.028848 | 0.990471 |
| 7. | DenseNet 121   (Augmented 2)     | 0.989993 | 0.03839  | 0.989993 |
| 8. | MobileNet V2 (Non   Augmented 1) | 0.988802 | 0.034398 | 0.988796 |

</div>
 
### Fruit Ripeness
 
#### Apple

<div align="center">
  
Tabel 12. Metrics of Apple Fruit Ripeness Object

| No |    Model     |   Accuracy   |     Loss     |      F1-Score      |
|:--:|:------------:|:------------:|:------------:|:------------:|
| 1. | Xception     | 0.9915611700 | 0.0292272900 | 0.9915598291 |
| 2. | MobileNet V2 | 0.9957805900 | 0.0181179500 | 0.9957802902 |
| 3. | DenseNet 121 | 0.9915611700 | 0.0274607200 | 0.9915598291 |
  
</div>

#### Banana

<div align="center">
  
Tabel 13. Metrics of Banana Fruit Ripeness Object

| No |    Model    |   Accuracy   |     Loss     |   F1-Score   |
|:--:|:-----------:|:------------:|:------------:|:------------:|
| 1. | Xception    | 0.9679525500 | 0.0881697500 | 0.967863693  |
| 2. | DenseNet 121 | 0.9727003000 | 0.0916466900 | 0.9722409617 |
  
</div>

#### Guava

<div align="center">
  
Tabel 14. Metrics of Guava Fruit Ripeness Object
 
| No |    Model    |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:-----------:|:----------:|:----------:|:----------:|
| 1. | Xception    | 1.00000000 | 0.00984316 | 1.00000000 |
| 2. | MobileNetV2 | 1.00000000 | 0.00380268 | 1.00000000 |
| 3. | DenseNet121 | 1.00000000 | 0.00990098 | 1.00000000 |
  
 </div>
 
 #### Lime
 
<div align="center">
  
Tabel 15. Metrics of Lime Fruit Ripeness Object
  
| No |    Model    |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:-----------:|:----------:|:----------:|:----------:|
| 1. | Xception    | 0.94827586 | 0.10563536 | 0.94718559 |
| 2. | MobileNetV2 | 0.98275864 | 0.04840786 | 0.98255056 |
| 3. | DenseNet121 | 0.99425286 | 0.03994407 | 0.99418352 |

</div>

#### Orange

<div align="center">
  
Tabel 16. Metrics of Orange Fruit Ripeness Object

| No |    Model    |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:-----------:|:----------:|:----------:|:----------:|
| 1. | Xception    | 0.97071129 | 0.10480809 | 0.97063727 |
| 2. | MobileNetV2 | 0.99581587 | 0.03788616 | 0.99580856 |
| 3. | DenseNet121 | 0.99163181 | 0.04456567 | 0.99160697 |

</div>

#### Pomegranate

<div align="center">
  
Tabel 17. Metrics of Pomegranate Fruit Ripeness Object

| No |    Model    |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:-----------:|:----------:|:----------:|:----------:|
| 1. | Xception    | 0.99371070 | 0.02224047 | 0.99266639 |
| 2. | MobileNetV2 | 1.00000000 | 0.01458145 | 1.00000000 |
| 3. | DenseNet121 | 0.99371070 | 0.02909623 | 0.99258292 |

</div>

### Vegetable Ripeness
#### Bell Pepper

<div align="center">
  
Tabel 18. Metrics of Bell Pepper Vegetable Ripeness Object

| No |            Model           |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:--------------------------:|:----------:|:----------:|:----------:|
| 1. | DenseNet 121 (Version   2) | 0.99380803 | 0.04155298 | 0.99184300 |
| 2. | MobileNetV2 (Version   1)  | 0.97832817 | 0.05807427 | 0.97176500 |
| 3. | MobileNetV2 (Version   2)  | 0.97213620 | 0.07268561 | 0.96369700 |
| 4. | DenseNet 121 (Version   1) | 0.96904027 | 0.08719029 | 0.95921600 |
| 5. | Xception (Version 2)       | 0.96284831 | 0.14397764 | 0.95147400 |
| 6. | Xception (Version 1)       | 0.94736844 | 0.15385775 | 0.93215000 |

</div>

#### Chile Pepper

<div align="center">
  
Tabel 19. Metrics of Chile Pepper Vegetable Ripeness Object

| No |            Model           |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:--------------------------:|:----------:|:----------:|:----------:|
| 1. | DenseNet 121 (Version   1) | 0.94841927 | 0.18507145 | 0.94173900 |
| 2. | DenseNet 121 (Version   2) | 0.94841927 | 0.16184105 | 0.94544700 |
| 3. | MobileNetV2 (Version   2)  | 0.93510813 | 0.27050444 | 0.92514100 |
| 4. | MobileNetV2 (Version   1)  | 0.92512476 | 0.25853795 | 0.91133100 |
| 5. | Xception (Version 2)       | 0.92179698 | 0.26123106 | 0.91241700 |
| 6. | Xception (Version 1)       | 0.91347754 | 0.26612681 | 0.90240000 |
  
</div>

#### Tomato

<div align="center">
  
Tabel 20. Metrics of Tomato Vegetable Ripeness Object

| No |            Model           |  Accuracy  |    Loss    |  F1-Score  |
|:--:|:--------------------------:|:----------:|:----------:|:----------:|
| 1. | DenseNet 121 (Version   2) | 0.97746480 | 0.13558786 | 0.97805800 |
| 2. | Xception (Version 2)       | 0.97746480 | 0.16184105 | 0.97753300 |
| 3. | MobileNetV2 (Version   2)  | 0.97183096 | 0.18801774 | 0.97225900 |
| 4. | Xception (Version 1)       | 0.97183096 | 0.20872307 | 0.97102600 |
| 5. | DenseNet 121 (Version   1) | 0.96901411 | 0.16311702 | 0.97000400 |
| 6. | MobileNetV2 (Version   1)  | 0.96901411 | 0.19508788 | 0.97062800 |
  
</div>

## Notes
The difference between "Augmented 1" and "Augmented 2" or "Version 1" and "Version 2" is layer used before the output layer. Models with keys "Augmented 1" and "Version 1" use GlobalMaxPooling2D() while on models with keys "Augmented 2" and "Version 2" use GlobalAveragePooling2D() form TensorFlow.

## References
[1] Suryawanshi, Yogesh; PATIL, Kailas; Chumchu, Prawit (2022), “VegNet: Vegetable Dataset with quality (Unripe, Ripe, Old, Dried and Damaged)”, Mendeley Data, V1, doi: 10.17632/6nxnjbn9w6.

[2] PATIL, Kailas; MESHRAM, Vishal (2021), “FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed quality)”, Mendeley Data, V2, doi: 10.17632/b6fftwbr2v.2

[3] Roboflow Universe Projects, "Banana Ripeness Classification Dataset," Roboflow Universe, Roboflow, Dec. 2022. [Online]. Available: https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification.

[4] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, "PlantDoc: A Dataset for Visual Plant Disease Detection," in Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, Hyderabad, India, 2020, pp. 249-253, doi: 10.1145/3371158.3371196.

[5] K. Team, "Keras documentation: Keras Applications," Keras.io, 2023. [Online]. Available: https://keras.io/api/applications/.
