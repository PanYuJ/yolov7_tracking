# yolov7-tracking
### 1. Environment
 - python=3.7.0 
 - pytorch=1.7.0
 - torchvision=0.8.0
 - cudatoolkit=11.0
 ```
!pip install -r requirements.txt
!pip install motmetrics
!pip install cython_bbox
!pip install lap
!pip install pycocotools
 ```
### 2. Dataset
In this case, Using two different dataset for training.
 - COCO dataset (only include car, bus and truck). train images 13016
 - [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) dataset (only include car, bus and truck). train images 6169

### 3. Training
The model training follows [yolov7](https://github.com/WongKinYiu/yolov7)  
```
# train p5 models
python train.py --workers 8 --device 0 --epochs 50 --batch-size 32 --data data/coco_custom.yaml --img 640 640 --cfg cfg/training/yolov7-tiny.yaml --weights 'yolov7-tiny.pt' --name yolov7 --hyp data/hyp.scratch_custom_tiny.yaml
```
### 4. Tracking
 - Creating four color area on hidden image to detect the color of vehicle center changing (to south and to north). The demo is shown as below.
 - Hidden image (Left)
 - main image (Right)  
 
 <img src="/images/color area.jpg" width="400"/>   <img src="/images/main image.jpg" width="400"/>
 
```
python tracker/tracker_custom.py --img_size 640 --tracker bytetrack --model_path your model weight --track_dataset Video  --dataset tracker/config_files/yolov7_track_custom.yaml
```
### 5. Demo
![gif](/images/highway-gif.gif)
