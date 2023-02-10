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
    import sys 
    sys.path.append(os.getcwd())   
    from models.experimental import attempt_load
    # from utils.datasets import LoadWebcam
    from utils.general import check_imshow
    from evaluate import evaluate
    from utils.torch_utils import select_device, time_synchronized, TracedModel
    print('Note: running yolo v7 detector')

except:
    pass

import tracker_dataloader
import trackeval

def set_basic_params(cfgs):
    global CATEGORY_DICT, DATASET_ROOT, CERTAIN_SEQS, IGNORE_SEQS, YAML_DICT
    CATEGORY_DICT = cfgs['CATEGORY_DICT']
    DATASET_ROOT = cfgs['DATASET_ROOT']
    CERTAIN_SEQS = cfgs['CERTAIN_SEQS']
    IGNORE_SEQS = cfgs['IGNORE_SEQS']
    YAML_DICT = cfgs['YAML_DICT']


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

    # NOTE: if save video, you must save image
    if opts.save_videos:
        opts.save_images = True

    """
    1. load model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ckpt = torch.load(opts.model_path, map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()  # for yolo v7

    # if opts.trace:
    #     print(opts.img_size)
    #     model = TracedModel(model, device, opts.img_size)
    # else:
    #     model.to(device)

    """
    2. load dataset and track
    """
    # ----------------sequence inference----------------
      # track per seq
      # firstly, create seq list
    if opts.track_dataset == 'seg':
      seqs = []
      if opts.data_format == 'yolo':
          with open(f'{DATASET_ROOT}/test.txt', 'r') as f:
              lines = f.readlines()
              
              for line in lines:
                  elems = line.split('/')  # devide path by / in order to get sequence name(elems[-2])
                  if elems[-2] not in seqs:
                      seqs.append(elems[-2])

      elif opts.data_format == 'origin':
          DATA_ROOT = os.path.join(DATASET_ROOT, '/sequences')
          seqs = os.listdir(DATA_ROOT)
      else:
          raise NotImplementedError
      seqs = sorted(seqs)
      seqs = [seq for seq in seqs if seq not in IGNORE_SEQS]
      print(f'Seqs will be evalueated, total{len(seqs)}:')
      print(seqs)

      # secondly, for each seq, instantiate dataloader class and track
      # every time assign a different folder to store results
      folder_name = strftime("%Y-%d-%m %H:%M:%S", gmtime())
      folder_name = folder_name[5:-3].replace('-', '_')
      folder_name = folder_name.replace(' ', '_')
      folder_name = folder_name.replace(':', '_')
      folder_name = opts.tracker + '_' + folder_name

      BaseTrack._count = 0

      # --------------tracking seq {seq}--------------
      for seq in seqs:

          path = os.path.join(DATA_ROOT, seq) if opts.data_format == 'origin' else os.path.join(f'{DATASET_ROOT}', 'test.txt')

          loader = tracker_dataloader.TrackerLoader(path, opts.img_size, opts.data_format, seq)

          data_loader = torch.utils.data.DataLoader(loader, batch_size=1)

          tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30, gamma=opts.gamma)  # instantiate tracker  TODO: finish init params

          results = []  # store current seq results
          frame_id = 0

          pbar = tqdm.tqdm(desc=f"{seq}", ncols=80)
          for i, (img, img0) in enumerate(data_loader):
              pbar.update()
              timer.tic()  # start timing this img
              
              if not i % opts.detect_per_frame:  # if it's time to detect
                current_tracks = tracker_inference(model, tracker, img, img0, device)
            
              # save results
              cur_tlwh, cur_id, cur_cls = [], [], []
              for trk in current_tracks:
                  
                  bbox = trk.tlwh
                  id = trk.track_id
                  cls = trk.cls

                  # filter low area bbox
                  if bbox[2] * bbox[3] > opts.min_area:
                      cur_tlwh.append(bbox)
                      cur_id.append(id)
                      cur_cls.append(cls)
                      # results.append((frame_id + 1, id, bbox, cls))

              results.append((frame_id + 1, cur_id, cur_tlwh, cur_cls))
              timer.toc()  # end timing this image
              
              if opts.save_images:
                  plot_img(img0, frame_id, [cur_tlwh, cur_id, cur_cls], save_dir=os.path.join(DATASET_ROOT, 'reuslt_images', seq))
              frame_id += 1

          seq_fps.append(i / timer.total_time)  # cal fps for current seq
          timer.clear()  # clear for next seq
          pbar.close()
          # thirdly, save results
          # every time assign a different name
          save_results(folder_name, seq, results)

          ## finally, save videos
          if opts.save_images and opts.save_videos:
              save_videos(seq_names=seq)
              
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
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened() :
            # time.sleep(1.0)
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
    
    if opts.track_dataset == 'Video':
        
        tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30, gamma=opts.gamma)
        if type(opts.img_size) == int:
            width, height = opts.img_size, opts.img_size
        elif type(opts.img_size) == list or type(opts.img_size) == tuple:
            width, height = opts.img_size[0], opts.img_size[1]
        
        BaseTrack._count = 0
        cap =cv2.VideoCapture(DATASET_ROOT)
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
                
                cur_tlwh, cur_id, cur_cls, cur_center = [], [], [], []
                for trk in current_tracks:
                    box = trk.tlwh # x_bottom_left, y_bottom_left, width, height
                    id = trk.track_id
                    
                    cls = trk.cls
                    
                # filter low area bbox
                    # if bbox[2] * bbox[3] > opts.min_area:
                    if box[2] * box[3] > opts.min_area:
                      # cur_tlwh.append(bbox)
                      # cur_id.append(id)
                      # cur_cls.append(cls)
                      # memory[cur_id[-1]] = cur_tlwh[-1]
                      memory[id] = box
                      
                # results = [cur_tlwh, cur_id, cur_cls]
                # tlwhs, ids, clss = results[0], results[1], results[2]

                    # Current frame
                # for i, (box, id, cls) in enumerate(zip(tlwhs, ids, clss)):
                    center = tuple([int(box[0]+(box[2]/2)), int(box[1]+(box[3]/2))])
                    
                    # draw a rect
                    cv2.rectangle(img_, [int(box[0]), int(box[1])], [int(box[0] + box[2]), int(box[1] + box[3])], get_color(id), thickness=1, )
                    cv2.circle(img_, center, 4, get_color(id), thickness=-1)
                    color_current = sub_img[center[1], center[0]]
                    
                    print('current_id:',id, color_current)
                    
                    # note the id and cls
                    text = f'{CATEGORY_DICT[cls]}-{id}'
                    cv2.putText(img_, text, [int(box[0]), int(box[1])], fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                                    color=get_color(id), thickness=1)
                
                    # previous frame
                    if id in previous:
                        previous_box = previous[id]
                        
                        # convert tlwh to tlbr
                        # previous_tlbr = tuple([int(previous_box[0]), int(previous_box[1]), int(previous_box[0] + previous_box[2]), int(previous_box[1] + previous_box[3])])
                        previous_center = tuple([int(previous_box[0]+(previous_box[2]/2)), int(previous_box[1]+int(previous_box[3]/2))])
                        
                        color_previous = sub_img[previous_center[1], previous_center[0]]
                        print('previous_id:',id, color_previous)
                        # UP_south = (color_current == [0,0,255]).all() and (color_previous == [0,255,0]).all()
                
                        DOWN_south = (color_previous == [0,0,255]).all() and (color_current==[0,255,0]).all()
                        
                        UP_north = (color_current == [0,0,254]).all() and (color_previous == [0,254,0]).all()
                        
                        # DOWN_north = (color_previous == [0,0,254]).all() and (color_current==[0,254,0]).all()
                        
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
                    
                    cv2.putText(img_, f'car: {car_south}\nbus: {bus_south}\ntruck: {truck_south}', (200, 500), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, 
                                    color=(0, 0, 255), thickness=4)
                    
                    cv2.putText(img_, f'car: {car_north}\nbus: {bus_north}\ntruck: {truck_north}', (500, 500), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, 
                                    color=(0, 255, 0), thickness=4)
                    
                    
                        # if CATEGORY_DICT[cls] =='car':
                        #     car +=1
                        # cv2.putText(img_, text, (box[0], box[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        #                 color=get_color(id), thickness=1)
                        
                        # cv2.putText(img_, text, (box[0], box[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        #                 color=get_color(id), thickness=1)
                        
                        # cv2.putText(img_, text, (box[0], box[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        #                 color=get_color(id), thickness=1)

                    # if int(center[0]) < 591 and int(center[1])<
                    # if cls==0 # 0:car
                    
                    # if cls ==1 # 1:bus
                    
                    # if cls ==2 # 2:truck
                    
                    # if center
                    
                    # if (time.time() - start_time) != 0:
                    #     cv2.putText(img_, "FPS{0}".format('%.1f' % (1/(time.time() - start_time))),
                    #                 (100,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
                
                cv2.imshow('',img_)
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
    
    # CounterLine1
    # cv2.line(img, (314,198+25), (591,249+25), (0, 0, 255), 50)
    # cv2.line(img, (314,198-25), (591,249-25), (0, 255, 0), 50)
    
    # # CounterLine2
    # cv2.line(img, (728,265+25), (1093,304+25), (0, 0, 254), 50)
    # cv2.line(img, (728,265-25), (1093,304-25), (0, 254, 0), 50)
    return img

        
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

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
            
        # if cls
            
    if show_image:    
      cv2.imshow('FRAME',img_)
      cv2.setMouseCallback('FRAME', POINTS)

def save_videos(seq_names):
    """
    convert imgs to a video
    seq_names: List[str] or str, seqs that will be generated
    """
    if not isinstance(seq_names, list):
        seq_names = [seq_names]

    for seq in seq_names:
        images_path = os.path.join(DATASET_ROOT, 'reuslt_images', seq)
        images_name = sorted(os.listdir(images_path))

        to_video_path = os.path.join(images_path, '../', seq + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        img0 = Image.open(os.path.join(images_path, images_name[0]))
        vw = cv2.VideoWriter(to_video_path, fourcc, 15, img0.size)

        for img in images_name:
            if img.endswith('.jpg'):
                frame = cv2.imread(os.path.join(images_path, img))
                vw.write(frame)
    print('Save videos Done!!')
    
def save_results(folder_name, seq_name, results, data_type='default'):
    """
    write results to txt file
    results: list  row format: frame id, target id, box coordinate, class(optional)
    to_file: file path(optional)
    data_type: write data format
    """
    assert len(results)
    if not data_type == 'default':
        raise NotImplementedError 

    if not os.path.exists(f'./tracker/results/{folder_name}'):

        os.makedirs(f'./tracker/results/{folder_name}')

    with open(os.path.join('./tracker/results', folder_name, seq_name + '.txt'), 'w') as f:
        for frame_id, target_ids, tlwhs, clses in results:
            if data_type == 'default':

                # f.write(f'{frame_id},{target_id},{tlwh[0]},{tlwh[1]},\
                #             {tlwh[2]},{tlwh[3]},{cls}\n')
                for id, tlwh, cls in zip(target_ids, tlwhs, clses):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{int(cls)}\n')
    f.close()

    return folder_name

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
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')
    # parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    
    # detect per several frames
    parser.add_argument('--detect_per_frame', type=int, default=1, help='choose how many frames per detect')    
    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')
   
    # args=['--dataset','config_files/yolov7_track_custom.yaml',
    #     '--data_format', 'yolo', 
    #     '--tracker', 'bytetrack',
    #     '--model_path', 'weights/yolov7_tiny_best.pt',
    #     '--img_size', '640',
    #     '--track_eval', 'False',
    #     '--track_dataset', 'Webcam'
    #     ]  
    opts = parser.parse_args()

    # NOTE: read path of datasets, sequences and TrackEval configs

    with open(opts.dataset, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    tracker_custom(opts, cfgs)