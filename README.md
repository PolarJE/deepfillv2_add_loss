# deepfillv2_add_loss

A tensorflow implementation for the paper [Deepfill v2 : Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589) (ICCV 2019 Oral). <br>
I added the perceptual loss and the style Loss to the original DeepFill v2 to reduce the artifacts of the image and improve the inpainting performance.

The perceptual loss and style loss computes the L1 distances but after projecting images into higher level feature spaces using an ImageNet-pretrained VGG-16. In the style loss, I first perform an autocorrelation (Gram matrix) on each feature map before applying the L1 distance.<br>
If you want to use the perceptual loss and style loss, download [imagenet-vgg-verydeep-19.mat](https://drive.google.com/file/d/15X7W90_3bcBK2PWxbU-8XI6In7MF5hFb/view?usp=sharing) first.

Requirement
-----------
- Python 3
- OpenCV-Python
- Numpy
- tensorflow
- tensorflow toolkit neuralgym (run `pip install git+https://github.com/JiahuiYu/neuralgym`).

Train
-----
- Prepare training images filelist and shuffle it
- Modify [inpaint.yml](https://github.com/PolarJE/deepfillv2_add_loss/blob/main/inpaint.yml) to set DATA_FLIST, LOG_DIR, IMG_SHAPES and other parameters.
- Run `python train.py`.

Test
----
- Run `python test.py --image folder_name/input.png --mask folder_name/mask.png --output folder_name/output.png --checkpoint model_logs/your_model_dir`.

Other resources
--------------
- [Original Deepfill v2 implementation](https://github.com/JiahuiYu/generative_inpainting)
- [Image Inpainting for Irregular Holes Using Partial Convolutions (Perceptual Loss, Style Loss)](https://arxiv.org/abs/1804.07723)
