# Naive UNet

Refer to

+ <a href="https://arxiv.org/pdf/1505.04597">Ronneberger et al.</a>
+ <a href="https://www.coursera.org/lecture/convolutional-neural-networks/semantic-segmentation-with-u-net-rEYzz">Andrew Ng's course</a>
+ <a href="https://www.kaggle.com/competitions/carvana-image-masking-challenge/overview">Dataset source</a>

For simplicity, some details are different from the original paper:

+ Both the input size and the output size are 128 $\times$ 128, which requires less parameters and is free of cropping.
+ The padding mode of convolution layer is `same`.

The result is sketchy, without fine-tuning , normalization or more training epochs. It only displayed how the network architechture works.
