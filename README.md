# Face-Identification
Models trained on the Labeled Faces in the Wild dataset on a face identification task. Added online semi-hard negative and hard negative triplet generation and a new method to resize negative sample generation to speed up training.
#
## Data preprocessing
Classes with too many images are reduced to 20. Classes with two images are expanded to 3 by augmentation, and classes with one image are not expanded.
#
Classes distribution before and after data processing.
![alt text](https://github.com/AlekseiZavalnyi/Face-identification/blob/main/images/class_distribution.png)



#

## Test models
Results on test data of the model with hard negative and semi-hard negative generation.
![alt text](https://github.com/AlekseiZavalnyi/Face-identification/blob/main/images/distribution_on_test_data.png)
![alt text](https://github.com/AlekseiZavalnyi/Face-identification/blob/main/images/test_scores.png)
