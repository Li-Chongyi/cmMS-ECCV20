# RGB-D Salient Object Detection with Cross-Modality Modulation and Selection



# TensorFlow
TensorFlow implementation of cmMS for RGBD salient object detection

## Requirements
```
Python 3
TensorFlow 1.x
```

### Folder structure
The following shows the basic folder structure.
```

├── checkpoint
│   ├── coarse_224 # A pre-trained checkpoint (coarse.model-2850)
│   │   ├── checkpoint 
│   │   ├── coarse.model-2850.data-00000-of-00001 
│   │   ├── coarse.model-2850.index 
│   │   ├── coarse.model-2850.meta
├── vgg_pretrained # vgg pretrained model 
│   ├── imagenet-vgg-verydeep-16
│   ├── imagenet-vgg-verydeep-19
│   ├── vgg16.npy
│   ├── vgg19.npy
├── main_test.py # testing code
├── model.py # network
├── test_real. # put RGB images here
├── depth_real. # put depth images here
├── ops.py.py
├── utils.py
├── vgg.py
```

### Preporcessing


1) download the pretrained VGG model
```
Google Drive: https://drive.google.com/open?id=15WlRLFSYG-mQ73DpUngaUqOnUwtz4PPc

Baidu Cloud: https://pan.baidu.com/s/1DVAwqe3n5JeUIuaAkzteYw  Password: byxj
```
2) download the checkpoint of our model and put it to 'checkpoint/coarse_224' folder
```
Google Drive: https://drive.google.com/open?id=15WlRLFSYG-mQ73DpUngaUqOnUwtz4PPc

Baidu Cloud: https://pan.baidu.com/s/1DVAwqe3n5JeUIuaAkzteYw  Password: byxj

```
3) normalize the depth maps (note that the foreground should have higher value than the background in our method)

2) resize the testing data to the size of 224*224

3) put your rgb images to 'test_real' folder and your depth maps to 'depth_real' folder (paired rgb image and depth map should have same name)


### Test
```
python main_test.py

find the results in the 'test_real' folder with the same name as the input image + "_out".

You can use a script to resize the results back to the same size as the original RGB-D image,  or just use the results with a size of 224*224 for evaluations. We did not find much differences for the evaluation results.
```

## Bibtex

```
@inproceedings{cmMS,
 author = {Li, Chongyi and Cong, Runmin and Piao Yongri and Xu Qianqian, and Loy, Chen Change},
 title = {RGB-D salient object eetection with cross-modality modulation and selection},
 booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
 pages    = {},
 month = {},
 year = {2020}
}
```

## Contact
If you have any questions, please contact Chongyi Li at lichongyi25@gmail.com.

## Our recent related work for RGB-D Salient Object Detection
https://github.com/Li-Chongyi/ASIF-Net
