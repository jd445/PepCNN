# PepCNN: Predict Peptide Detectability Based on Large Kernel Convolutional Neural Network
Source code of PepCNN, focusing on predicting peptide detectability.


# Abstract
Peptide detectability information is critical to many bioinformatics problems in proteomics. Currently, many methods based on deep learning have been proposed to predict peptide detectability. However, existing methods have high model complexities and are prone to overfitting during training. Here we propose the PepCNN, a large kernel convolutional neural network with a simple and efficient adaptive feature extraction encoder, based only on peptide sequences. In particular, we introduce the concept of gap constraints into the convolution operation via a specific large kernel. A mix-up strategy was also used for data augmentation. Experimental results show that our model performs ahead of the state-of-the-art approaches on the benchmark datasets for two species and its model complexity is much smaller than those state-of-the-art approaches.
