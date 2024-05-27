# Equiangular-Basis-Vectors

Paper is available at: http://arxiv.org/abs/2303.11637

Data for the 10w classes classification task can be download from: https://github.com/aassxun/Equiangular-Basis-Vectors/releases/download/10w_classes/10w_classes.zip  

It contains 120w images for training and 60w images for validation. (Resize to 224*224)

# Introduction
The initial problem addressed in this paper is the classification of large categories, such as categorizing 100,000 or even 1,000,000 classes. In networks like ResNet-50, the last linear layer would require 2048×100,000 or 2048×1,000,000 parameters, which would significantly exceed the parameter count of the preceding feature extraction layers.

On the other hand, typical classification problems use one-hot vectors as labels, which can be understood as orthogonal bases with an angle of 90 degrees between any two vectors. In 2022, the Annals of Mathematics published an article stating that as the dimension D approaches infinity, the number of lines with a given angle increases linearly with D (refer to "Equiangular lines with a fixed angle"). Therefore, if the angles are entirely equal, a large number of categories would require a very high dimension D (though such datasets are not common).

The idea of this paper was to optimize the angles so that when constrained to around 83-97 degrees (axially symmetric), 5,000 dimensions could accommodate 100,000 categories without significantly affecting classification performance. The corresponding dataset is also available as open source on GitHub. Additionally, when the angle is 0, there can be infinitely many such basis vectors in space, ensuring feasibility. However, regarding the relationship between alpha, the space dimension, and the number of such vectors, there is no fixed mathematical solution. In some special cases, solutions exist, and for those interested, please refer to ``Sparse and Redundant Representations From Theory to Applications in Signal and Image Processing" for further reading.

# Environment
mindspore 1.9

Local training can be quickly started using the code in 'Run_with_GPU'.
