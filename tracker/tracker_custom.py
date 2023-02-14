import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import cv2 
from PIL import Image
import tqdm

import argparse
import time
from time import gmtime, strftime
from timer import Timer
import yaml

from basetrack import BaseTracker, BaseTrack # for framework
from deepsort import DeepSORT
from bytetrack import ByteTrack
from deepmot import DeepMOT
from botsort import BoTSORT
from uavmot import UAVMOT
from strongsort import StrongSORT

try:  # import package that outside the tracker folder  For yolo v7
    sys.path.append(os.getcwd())   
    # from utils.datasets import LoadWebcam
    print('Note: running yolo v7 detector')

except:
    pass


def set_basic_params(cfgs):
    global CATEGORY_DICT, DATASET_ROOT
    CATEGORY_DICT = cfgs['CATEGORY_DICT']
    DATASET_ROOT = cfgs['DATASET_ROOT']

timer = Timer()
seq_fps = []  # list to store time used for every seq
def tracker_custom(opts, cfgs):
    set_basic_params(cfgs)  # NOTE: set basic path and seqs params first

    TRACKER_DICT = {
        'sort': BaseTracker,
        'deepsort': DeepSORT,
        'bytetrack': ByteTrack,
        'deepmot': DeepMOT,
        'botsort': BoTSORT,
        'uavmot': UAVMOT, 
        'strongsort': StrongSORT, 
    }  # dict for trackers, key: str, value: class(BaseTracker)

    # NOTE: ATTENTION: make kalman and tracker compatible
    if opts.tracker == 'botsort':
        opts.kalman_format = 'botsort'
    elif opts.tracker == 'strongsort':
        opts.kalman_format = 'strongsort'

    """
    1. load model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ckpt = torch.load(opts.model_path, map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()  # for yolo v7

    """
    2. load dataset and track
    """

    # ----------------Webcam inference----------------
    if opts.track_dataset == 'Webcam':
        tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30, gamma=opts.gamma)
        
        if type(opts.img_size) == int:
            width, height = opts.img_size, opts.img_size
        elif type(opts.img_size) == list or type(opts.img_size) == tuple:
            width, height = opts.img_size[0], opts.img_size[1]
        
        BaseTrack._count = 0
        cap =cv2.VideoCapture(0)
        start_time = time.time()
        fps_counter = 0
        
        
        while cap.isOpened() :
            success, im0s = cap.read()
            
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            key = cv2.waitKey(1) & 0xff
            if key == ord(' '):
                cv2.waitKey(0)
            if key == ord('q'):
                break
            
            fps_counter += 1
            
            img = cv2.resize(np.copy(im0s), (width, height))          
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0          
            img = torch.from_numpy(img)
            im0s = torch.from_numpy(im0s)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            frame_id = 0  
            current_tracks = tracker_inference(model, tracker, img, im0s, device, fps_counter)

            cur_tlwh, cur_id, cur_cls, cur_center = [], [], [], []
            for trk in current_tracks:
                bbox = trk.tlwh # x_bottom_left, y_bottom_left, width, height
                center = [bbox[0]+(width/2), bbox[1]+(height/2)]
                id = trk.track_id
                cls = trk.cls

            # filter low area bbox
                if bbox[2] * bbox[3] > opts.min_area:
                  cur_tlwh.append(bbox)
                  cur_id.append(id)
                  cur_cls.append(cls)
                  cur_center.append(center)

            plot_img(im0s, frame_id, [cur_tlwh, cur_id, cur_cls], start_time=start_time, fps_counter=fps_counter , save_dir=None, show_image=True)
            start_time = time.time()
            frame_id += 1
            fps_counter = 0
            
        cap.release()
        cv2.destroyAllWindows()
        
    # ----------------video inference----------------
    
    if opts.track_dataset == 'video':
        
        tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30, gamma=opts.gamma)
        if type(opts.img_size) == int:
            width, height = opts.img_size, opts.img_size
        elif type(opts.img_size) == list or type(opts.img_size) == tuple:
            width, height = opts.img_size[0], opts.img_size[1]
            
        BaseTrack._count = 0
        cap =cv2.VideoCapture(DATASET_ROOT)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        name = os.path.split(DATASET_ROOT)[1]
        name = name.split('.')[0] + '.mp4'
        vid_path = os.path.join(opts.save_path,name)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

        start_time = time.time()
        fps_counter = 0
        memory = {}
        car_south, bus_south, truck_south = 0, 0, 0
        car_north, bus_north, truck_north = 0, 0, 0
        while True:
            success, im0s = cap.read()
            
            if success:
                key = cv2.waitKey(1) & 0xff
                if key == ord(' '):
                    cv2.waitKey(0)
                if key == ord('q'):
                    break
                fps_counter += 1
                
                img = cv2.resize(np.copy(im0s), (width, height))          
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0          
                img = torch.from_numpy(img)
                im0s = torch.from_numpy(im0s)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                    
                frame_id = 0  
                current_tracks = tracker_inference(model, tracker, img, im0s, device, fps_counter)
                
                img_ = np.ascontiguousarray(np.copy(im0s))
                
                cv2.line(img_, (314,198), (591,249), (0, 0, 255), 5)
                cv2.line(img_, (728,265-40), (1093,304-40), (0, 255, 0), 5)
            
                sub_img = np.ascontiguousarray(np.copy(im0s))
                sub_img = draw_Counter(sub_img)
                cv2.imshow('123',sub_img)
                previous = memory.copy()
                memory = {}
                
                for trk in current_tracks:
                    box = trk.tlwh # x_bottom_left, y_bottom_left, width, height
                    id = trk.track_id
                    cls = trk.cls
                    
                    # filter low area bbox
                    if box[2] * box[3] > opts.min_area:
                      memory[id] = box
                      
                    # Current frame
                    center = tuple([int(box[0]+(box[2]/2)), int(box[1]+(box[3]/2))])
                    
                    # draw a rect
                    cv2.rectangle(img_, [int(box[0]), int(box[1])], [int(box[0] + box[2]), int(box[1] + box[3])], get_color(id), thickness=1, )
                    cv2.circle(img_, center, 4, get_color(id), thickness=-1)
                    color_current = sub_img[center[1], center[0]]
                    
                    # print('current_id:',id, color_current)
                    
                    # note the id and cls
                    text = f'{CATEGORY_DICT[cls]}-{id}'
                    cv2.putText(img_, text, [int(box[0]), int(box[1])], fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                                    color=get_color(id), thickness=1)
                    
                    # previous frame
                    if id in previous:
                        previous_box = previous[id]
                    
                        previous_center = tuple([int(previous_box[0]+(previous_box[2]/2)), int(previous_box[1]+int(previous_box[3]/2))])
                        
                        color_previous = sub_img[previous_center[1], previous_center[0]]
                        # print('previous_id:',id, color_previous)

                        DOWN_south = (color_previous == [0,0,255]).all() and (color_current==[0,255,0]).all()
                        UP_north = (color_current == [0,0,254]).all() and (color_previous == [0,254,0]).all()
                        
                        if DOWN_south:
                            if CATEGORY_DICT[cls]=='car':
                                car_south += 1
                            if CATEGORY_DICT[cls]=='bus':
                                bus_south += 1
                            if CATEGORY_DICT[cls]=='truck':
                                truck_south += 1
                                
                        if UP_north:
                            if CATEGORY_DICT[cls]=='car':
                                car_north += 1
                            if CATEGORY_DICT[cls]=='bus':
                                bus_north += 1
                            if CATEGORY_DICT[cls]=='truck':
                                truck_north += 1
                    
                    cv2.putText(img_, f'car: {car_south}', (200, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, 
                                    color=(0, 0, 255), thickness=4)
                    
                    cv2.putText(img_, f'bus: {bus_south}', (200, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, 
                                    color=(0, 0, 255), thickness=4)
                    
                    cv2.putText(img_, f'truck: {truck_south}', (200, 150), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, 
                                    color=(0, 0, 255), thickness=4)
                    
                    cv2.putText(img_, f'car: {car_north}', (1120, 200), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, 
                                    color=(0, 255, 0), thickness=4)
                    
                    cv2.putText(img_, f'bus: {bus_north}', (1120, 250), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, 
                                    color=(0, 255, 0), thickness=4)
                    
                    cv2.putText(img_, f'truck: {truck_north}', (1120, 300), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, 
                                    color=(0, 255, 0), thickness=4)

                if not opts.nosave:
                    if not os.path.exists(opts.save_path):
                        os.mkdir(opts.save_path)
                    vid_writer.write(img_)
      
                cv2.imshow('',img_)
                
                #Show X, Y coordinate on image
                # cv2.setMouseCallback('', POINTS)

                start_time = time.time()
                frame_id += 1
                fps_counter = 0
            else:
                break
            
        cap.release()
        cv2.destroyAllWindows()
        
def draw_Counter(img):
    # CounterPolylines1
    pts_s1 = np.array([[314, 198], [591, 249], [717, 176], [540, 122]], np.int32)
    pts_s2 = np.array([[314, 198], [591, 249], [398, 352], [210, 242]], np.int32)
    pts_n1 = np.array([[728, 265-40], [1093, 304-40], [1113, 163-40], [877, 140-40]], np.int32)
    pts_n2 = np.array([[728, 265-40], [1093, 304-40], [1035, 502-40], [540, 434-40]], np.int32)
    
    # CounterPolylines1
    cv2.fillPoly(img, [pts_s1], (0, 0, 255))
    cv2.fillPoly(img, [pts_s2], (0, 255, 0))
    
    # CounterPolylines2
    cv2.fillPoly(img, [pts_n1], (0, 0, 254))
    cv2.fillPoly(img, [pts_n2], (0, 254, 0))
    
    return img

def plot_img(img, frame_id, results, start_time, fps_counter, save_dir=None, show_image=False, ):
    """
    img: np.ndarray: (H, W, C)
    frame_id: int
    results: [tlwhs, ids, clses]
    save_dir: sr
    plot images with bboxes of a seq
    """

    if save_dir !=None:
      if not os.path.exists(save_dir):
          os.makedirs(save_dir)

    img_ = np.ascontiguousarray(np.copy(img))

    tlwhs, ids, clses = results[0], results[1], results[2]
    for tlwh, id, cls in zip(tlwhs, ids, clses):
        
        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        
        # draw a rect
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=1, )
        
        # note the id and cls
        text = f'{CATEGORY_DICT[cls]}-{id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=get_color(id), thickness=1)
        
        if (time.time() - start_time) != 0:
            cv2.putText(img_, "FPS{0}".format('%.1f' % (1/(time.time() - start_time))),
                        (100,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
 
def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    
    return color

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        
def tracker_inference(model, tracker, img, img0, device, fps_counter):
    
    if not fps_counter % opts.detect_per_frame:
        with torch.no_grad():
            out = model(img.to(device))  # model forward             
            out = out[0]  # NOTE: for yolo v7
          
        if len(out.shape) == 3:  # case (bs, num_obj, ...)
            # out = out.squeeze()
            # NOTE: assert batch size == 1
            out = out.squeeze(0)
            # img0 = img0.squeeze(0)
    
        # remove some low conf detections
        out = out[out[:, 4] > 0.001]
      
        # NOTE: yolo v7 origin out format: [xc, yc, w, h, conf, cls0_conf, cls1_conf, ..., clsn_conf]
        if opts.det_output_format == 'yolo':
            if len(out)==0:
                print('Error')
            else:
                cls_conf, cls_idx = torch.max(out[:, 5:], dim=1)
                # out[:, 4] *= cls_conf  # fuse object and cls conf
                out[:, 5] = cls_idx
                out = out[:, :6]
        
        current_tracks = tracker.update(out, img0)
    else:
        if len(img0.shape) == 4:
            img0 = img0.squeeze(0)
        current_tracks = tracker.update_without_detection(None, img0)
    return current_tracks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='visdrone', help='visdrone, or mot')
    parser.add_argument('--data_format', type=str, default='origin', help='format of reading dataset')
    parser.add_argument('--det_output_format', type=str, default='yolo', help='data format of output of detector, yolo or other')

    parser.add_argument('--tracker', type=str, default='bytetrack', help='sort, deepsort, etc')

    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')
    parser.add_argument('--img_size', type=int, default=800, help='[train, test] image sizes')
    parser.add_argument('--save_path', type=str, default='./results', help='save root')

    """For tracker"""
    # model path
    parser.add_argument('--reid_model_path', type=str, default='tracker/weights/ckpt.t7', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='tracker/weights/DHN.pth', help='path of DHN path for DeepMOT')

    # threshs
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='filter tracks')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU thresh to filter tracks')

    # other options
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--kalman_format', type=str, default='default', help='use what kind of Kalman, default, naive, strongsort or bot-sort like')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')
    parser.add_argument('--track_dataset', type=str, default='seg', help='input source mode (video, Webcam, image)')
    parser.add_argument('--nosave', action='store_true', help='save result')
    # parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    # parser.add_argument('--save_gif', action='store_true', help='save tracking results (gif)')
    
    # detect per several frames
    parser.add_argument('--detect_per_frame', type=int, default=1, help='choose how many frames per detect')    
    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')
   
    opts = parser.parse_args()


    with open(opts.dataset, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    tracker_custom(opts, cfgs)
