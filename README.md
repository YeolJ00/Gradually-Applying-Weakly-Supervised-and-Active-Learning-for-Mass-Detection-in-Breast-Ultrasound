## Gradually Applying Weakly Supervised and Active Learning for Mass Detection in Breast Ultrasound Images

This is the implementation for the paper "Gradually Applying Weakly Supervised and Active Learning for Mass Detection in Breast Ultrasound Images" (https://www.mdpi.com/2076-3417/10/13/4519)

The code is based on the Pytorch 1.0 implementation for Faster-RCNN (https://github.com/jwyang/faster-rcnn.pytorch)

## Preparation

First of all, clone the code

Then, create a folder:
```
cd faster-rcnn.pytorch && mkdir data
```

SNUBH dataset is not available for public due to personal information issues.

### prerequisites

* Python 2.7 or 3.6
* Pytorch 1.0 (for Pytorch 0.4.0 go to master branch)
* CUDA 8.0 or higher

### Pretrained Model

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the data/pretrained_model/.

**NOTE**. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
python setup.py build develop
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

### Train

The following command is an example of the train parameters we used to train our model

```
CUDA_VISIBLE_DEVICES=0 python trainval_net_alter.py --net vgg16 --cuda --s 9 --max_iter 160000 --lr 0.005 --gamma_for_alpha 16 --cag
```

### Test

To test the model, check the session, checkepoch, checkpoint =0

```
python test_net.py --net vgg16 --checksession 13 --checkepoch 160000 --checkpoint 0 --cuda --vis
```

Please check the argument code for detailed specifications of the model.



## Citation

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }