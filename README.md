# Fabric defect recognition

All of the code package and pre-trained models included with the paper "Fabric defect recognition using optimized neural network".

This paper utilizes a typical Convolutional Neural Network model that can be applied to recognize fabric defects with complex background, so as to solve the problem that complicated texture fabrics can not be detected effectively at present. Though the Convolutional Neural Networks are very powerful, the large number of parameters consumes considerable computation and memory bandwidth. In real world applications, however, the fabric defect recognition task needs to be carried out in a timely fashion on a computation limited platform. To optimize the deep Convolutional Neural Network, a novel way was introduced to reveal what input pattern originally caused a specific activation in the network feature maps. Using this visualization technique, this paper takes the feature visualizing of a representative fabric defect image as an example, and attempt to change the architecture of original neural network in order to reduce the computational load of defect detection system, and be more conducive to the transplantation of the small system. After a series of improvements, a new Convolutional Network has been acquired which is more conducive to the fabric images feature extraction, while the computational complexity and the total number of parameters only respectively accounts for 23% and 8.9% of the original model.

![LZFnets](https://github.com/ZCmeteor/Fabric-defect-recognition-/blob/master/experimental%20result/AlexNet-crack01/conv1/deconvolution/grid_image.png)
![LZFnets](https://github.com/ZCmeteor/Fabric-defect-recognition-/blob/master/experimental%20result/AlexNet-crack01/conv3/deconvolution/grid_image.png)



# The two screenshots below are representative recognition result.

![LZFnets](https://github.com/ZCmeteor/Fabric-defect-recognition-/blob/master/101.PNG)

![LZFnets](https://github.com/ZCmeteor/Fabric-defect-recognition-/blob/master/103.PNG)
