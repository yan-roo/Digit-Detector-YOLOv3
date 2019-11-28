#  Street View House Numbers Detection by Keras-YOLOv3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
<img src="output/1.png"><img src="output/6.png"><img src="output/7.png"><img src="output/9.png">
## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5

python yolo_video.py --model_path logs/svhn_weights/trained_weights_final.h5
									--anchors_path model_data/yolo_anchors.txt
									--classes_path model_data/svhn_classes.txt
									--gpu_num 1
									--input video_path

python yolo_video.py --img ./data/test/1.png
python yolo_video.py --imgdir ./data/test 
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

---
## Training

1. Download training data from [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/)   (*Format 1: Full Numbers*)
```
	wget http://ufldl.stanford.edu/housenumbers/train.tar.gz
	wget http://ufldl.stanford.edu/housenumbers/test.tar.gz
	tar zxvf train.tar.gz
	tar zxvf test.tar.gz
```

2. Annotation file and class names file **(option)**
	- Get SVHN VOC Annotations from [penny4860/svhn-voc-annotation-format](https://github.com/penny4860/svhn-voc-annotation-format)
	* Use **voc_annotation.py** to convert VOC annotations to txt version
	* Class names file is on **model_data/svhn_classes.txt**

	```
	wget https://github.com/penny4860/svhn-voc-annotation-format/archive/master.zip
unzip master.zip
python voc_annotation.py
	```

3. Generate your own annotation file and class names file. **(option)** 
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  `python train.py` 
	Or use Juypter Notebook `train_notebook_version.ipynb`
	
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify `annotation path`, `anchor path`, `classes path` ,  `log_dir path` and `weights_path` when create model.


---

### Usage
Use --help to see usage of yolo_video.py:
```
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
  --imgdir           Image dir detection mode, will ignore all positional arguments
  --txt                 Image dir detection will output txt files
```
---



## Some issues to know

1. The test environment is
    - Python 3.6.9
    - Keras 2.1.5
    - tensorflow-gpu 1.14.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.