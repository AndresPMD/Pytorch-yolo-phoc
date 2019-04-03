import sys
import os
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from shutil import copyfile, move

import struct # get_image_size
import imghdr # get_image_size

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0) # XMIN OF PREDICTION AND GROUND TRUTH
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0) # XMAX OF PREDICTION AND GROUND TRUTH
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0) # SAME WITH YMIN AND YMAX:
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
    # RECEIVES BOXES LIST, WITH DIMENSION LIST OF BOXES ( X * Y *W * H * PHOC)
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)): # 1 - CONFIDENCE TO SORT ?
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]] # BOXES WITH BETTER CONFIDENCES FIRST
        if box_i[4] > 0: # IF CONFIDENCE BIGGER THAN  0 APPEND
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)): # COMPARE IOU, IF IT IS BIGGER THAN THE NMS_THRESHOLD MAKE THE LESS CONFIDENT 0
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0

    return out_boxes

def sort_phocs(boxes, nms_thresh):
    # Define number of max proposed PHOCS
    max_phocs = 10
    if len(boxes) == 0:
        return boxes
    # RECEIVES BOXES LIST, WITH DIMENSION LIST OF BOXES ( X * Y *W * H * PHOC)
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)): # 1 - CONFIDENCE TO SORT
        det_confs[i] = 1-boxes[i][4]

    _,sortIds = torch.sort(det_confs)
    out_boxes = []

    for i in range(len(boxes)):
        if i <= max_phocs: # MAX NUMBER OF PROPOSED PHOCS
            box_i = boxes[sortIds[i]]
            if box_i[4] > 0:
                out_boxes.append(box_i)
        else:
            break

    return out_boxes

    '''
        box_i = boxes[sortIds[i]] # BOXES WITH BETTER CONFIDENCES FIRST
        if box_i[4] > 0: # IF CONFIDENCE BIGGER THAN  0 APPEND
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)): # COMPARE IOU, IF IT IS BIGGER THAN THE NMS_THRESHOLD MAKE THE LESS CONFIDENT 0
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    '''


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    # *** GETS ALL BOXES ABOVE A THRESHOLD

    anchor_step = len(anchors)/num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []
    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)

    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    # cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data

    # GETS THE PHOC REPRESENTATION:
    cls_confs = torch.nn.Sigmoid()(Variable(output[5:5 + num_classes].transpose(0, 1))).data ## START THINKING FROM HERE:

    # cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    # cls_max_confs = cls_max_confs.view(-1)
    # cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()
    
    # cls_max_confs = convert2cpu(cls_max_confs)
    # cls_max_ids = convert2cpu_long(cls_max_ids)
    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    # For PHOC Vector representation operations
    cls_confs = convert2cpu(cls_confs.view(-1, num_classes))

    ''' for valid.py CHECK:
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    '''

    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    conf = det_confs[ind]
                    '''
                    ****  TO BE CHECKED FOR VALIDATION/DETECT
                    if only_objectness:
                        conf =  det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
                    '''
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        # cls_max_conf = cls_max_confs[ind]
                        # cls_max_id = cls_max_ids[ind]
                        # box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]

                        cls_phoc = cls_confs[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_phoc]

                        '''
                        *** TO BE CHECKED FOR VALIDATION SCRIPT VALID.PY
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        '''
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2]/2.0) * width))
        y1 = int(round((box[1] - box[3]/2.0) * height))
        x2 = int(round((box[0] + box[2]/2.0) * width))
        y2 = int(round((box[1] + box[3]/2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img

def plot_boxes(img, boxes, words, neigh, savename=None, class_names=None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)

    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 6 and class_names:
            cls_conf = box[4]
            cls_id = box[5]

            result = neigh.predict(cls_id.unsqueeze(0).numpy())
            text_found = words[result[0]]


            #print('%s: %f' % (text_found, cls_conf))
            classes = len(class_names)
            offset = 2*i * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)

            fontsize = height/45
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", fontsize)
            draw.text((x1, y1), text_found, fill=rgb, font= font)
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img

def count_boxes(img, boxes, destination_folder):
    # If there is at least one box (some text in image) copy the image to a destination folder
    if len(boxes) >= 1:
        print("Text found on %s, saving image!" % img)
        copyfile(img, destination_folder+img.split('/')[-1].strip())
    return

def write_text_result_icdar(img, boxes, words, neigh, savename=None, class_names=None):
    # ******** RESULTS FOR ICDAR 13/15 FORMAT *****
    width = img.width
    height = img.height

    if savename:
        path_list = savename.split('/')
        img_name = path_list[-1]
        path_string = '/'.join(path_list[:-1])+'/'
        result_file = open(path_string+'res_'+ img_name[:-4] + '.txt', 'w')
        print("Saving results to ", path_string+'res_'+ img_name[:-4] + '.txt', 'w')


    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        # PROCESS COORDINATES TO WRITE ACCORDING TO ICDAR 13 FILE FORMAT
        # - Clockwise xi,yi describe the 4 points
        p1x, p1y = int(x1), int(y1)
        p2x, p2y = int(x2), int(y1)
        p3x, p3y = int(x2), int(y2)
        p4x, p4y = int(x1), int(y2)


        if len(box) >= 6 and class_names:
            cls_conf = box[4]
            cls_id = box[5]

            result = neigh.predict(cls_id.unsqueeze(0).numpy())
            text_found = words[result[0]]

            result_file.write(str(p1x) + ',' + str(p1y) + ',' + str(p2x) + ',' + str(p2y) + ',' + str(p3x) + ',' + str(p3y) + ',' + str(p4x) + ',' + str(p4y) + ',' + str(text_found) + '\n')
            #print('%s: %f' % (text_found, cls_conf))

    result_file.close()

    return result_file

def write_text_result(img, boxes, words, neigh, savename=None, class_names=None):
    # BIGGEST BBOX FIRST USED FOR VQA
    width = img.width
    height = img.height

    if savename:
        result_file = open(savename.replace('jpg','txt').replace('JPEG','txt').replace('JPG','txt'), 'w')
        #print("Saving results to: ", result_file)
    areas = []
    text = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        # PROCESS COORDINATES TO WRITE ACCORDING TO ICDAR 13 FILE FORMAT
        # - Clockwise xi,yi describe the 4 points
        p1x, p1y = int(x1), int(y1)
        p2x, p2y = int(x2), int(y1)
        p3x, p3y = int(x2), int(y2)
        p4x, p4y = int(x1), int(y2)

        area_box = [(x2-x1)*(y2-y1)]
        areas.append(area_box)

        if len(box) >= 6 and class_names:
            cls_conf = box[4]
            cls_id = box[5]

            result = neigh.predict(cls_id.unsqueeze(0).numpy())
            text_found = words[result[0]]
            text.append(text_found)

    ordered_text = [x for _, x in sorted(zip(areas, text), reverse= True)]
    for i in range(len(text)):
        result_file.write(str(ordered_text[i]) + '\n')

    result_file.close()

    return

def write_retrieval (img, boxes, savename=None, class_names=None):

    width = img.width
    height = img.height

    if savename:
        #print("save results to %s" % savename.replace('jpg','txt'))
        result_file = open(savename[:-4] + '.txt', 'w')
        #result_file = open(savename + '.txt', 'w')


    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        if len(box) >= 6 and class_names:
            cls_conf = box[4]
            cls_id = box[5]
            np.set_printoptions(suppress=True)
            phoc_retrieved = (cls_id.unsqueeze(0).numpy())

            result_file.write(str(phoc_retrieved) + '\n')
            #print('%s: %f' % (text_found, cls_conf))

    result_file.close()

    return result_file


def move_image(img_full_path, processed_path):

    filename = img_full_path.split('/')[-1]
    move(img_full_path, processed_path+filename)
    return

def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        text_file = open(lab_path, 'r')
        lines = text_file.read().split('\n')
        del lines[-1]
        truths = list()
        for k in range(len(lines)):
            truths.append(lines[k].split(" "))
        # truths = np.loadtxt(lab_path)
        # truths = truths.reshape(truths.size/5, 5) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    words = []
    for i in range(len(truths)):
        if truths[i][3] < min_box_scale:
            continue
        words.append(truths[i][0])
        new_truths.append([float(0), float(truths[i][1]), float(truths[i][2]), float(truths[i][3]), float(truths[i][4])])
    return np.array(new_truths), words

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    output = model(img)
    output = output.data
    #for j in range(100):
    #    sys.stdout.write('%f ' % (output.storage()[j]))
    #print('')
    t3 = time.time()

    # GET BOXES IN DETECTION
    # get most probable phocs DEFINED INSIDE THIS FUNCTION

    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)[0]
    #for j in range(len(boxes)):
    #    print(boxes[j])
    t4 = time.time()

    # MODIFIED FOR VQA - ORIGINAL DOES NOT REQUIRE SORT PHOCS
    # boxes = sort_phocs(boxes, nms_thresh)
    boxes = nms(boxes, nms_thresh)


    t5 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('             nms : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t0))
        print('-----------------------------------')
    return boxes

def do_detect_retrieval(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    output = model(img)
    output = output.data
    #for j in range(100):
    #    sys.stdout.write('%f ' % (output.storage()[j]))
    #print('')
    t3 = time.time()

    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)[0]
    #for j in range(len(boxes)):
    #    print(boxes[j])
    t4 = time.time()

    # get most probable phocs DEFINED INSIDE THIS FUNCTION
    boxes = sort_phocs(boxes, nms_thresh)
    t5 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('             nms : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t0))
        print('-----------------------------------')
    return boxes

def do_feature_extraction(model, img, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray:
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)

    output = model(img)

    # Transform to numpy
    output = output.data.cpu().numpy()[0]

    output = output.transpose(1,2,0)

    return output

def do_select_phocs(output, use_cuda =1):
    # Selects the most confident PHOCs per grid cell and reshape the output as B x W x H x (C+604 (PHOC))
    output = output.reshape((19, 19, 13, 609))
    w, h, anchors, _ = np.shape(output)

    conf_matrix = output[:, :, :, 4]
    # SIGMOID OF CONFIDENCES
    conf_matrix = 1 / (1 + np.exp(-conf_matrix))

    # SIGMOID TO OBTAIN THE PHOCS
    phoc_matrix = output[:, :, :, 5:]
    phoc_matrix = 1 / (1 + np.exp(-phoc_matrix))

    new_matrix = np.zeros((19,19,605))

    for i in range (w):
        for j in range(h):
            for k in range (0,13):
                if conf_matrix[i][j][k] > new_matrix[i][j][0]:
                    new_matrix[i][j][0] = conf_matrix[i][j][k]
                    new_matrix[i][j][1:] = phoc_matrix[i][j][k][:]
    return new_matrix



def KNNclassifier():
    print('Loading dictionary...')

    '''
     # 1K MAFLA SYNTH COLOR DATASET DICTIONARY
    with open("/media/amafla/ssd/maflaflow/dictionaries/1k_words_synth_AM/words.txt", "rb") as fp:
        words = pickle.load(fp)

    with open("/media/amafla/ssd/maflaflow/dictionaries/1k_words_synth_AM/phocs.txt", "rb") as fp:
        phocs = pickle.load(fp)
    '''

     #  JADERBERG 90K + IAM DATASET DICTIONARY
    with open("/home/amafla/Documents/maflaflow/dictionaries/Jaderberg_IAM_py27/words.txt", "rb") as fp:
        words = pickle.load(fp)

    with open("/home/amafla/Documents/maflaflow/dictionaries/Jaderberg_IAM_py27/phocs.txt", "rb") as fp:
        phocs = pickle.load(fp)

    # USE FULL DICTIONARY:
    y = np.arange(0, (len(words))-1, 1)

    # USE MORE COMMON WORDS:
    # y = np.arange(0,10000,1)

    neigh = KNeighborsClassifier(n_neighbors=1)


    # IAM DICTIONARY:
    neigh.fit(phocs[0:88623], y)
    print('Dictionary loaded and KNN trained...!')


    #return words, phocs, neigh
    return words, neigh

def read_data_cfg(datacfg):
    '''
    Returns in options the data cfg file info
    :param datacfg: path with data cfg
    :return: options
    '''
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets
      
def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close( )
    return count

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24: 
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2 
                ftype = 0 
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2 
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
