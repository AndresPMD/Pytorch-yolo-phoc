import sys
import time
import os
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

def detect(cfgfile, weightfile, imgfolder):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/recognition.names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    class_names = load_class_names(namesfile)
    image_list = os.listdir(imgfolder)

    # words, neigh = KNNclassifier()

    for imgfile in image_list:
        img_full_path = imgfolder+imgfile
        img = Image.open(img_full_path).convert('RGB')
        sized = img.resize((m.width, m.height))
        conf_threshold = 0.35
        nms_threshold = 0.5
        for i in range(2):
            start = time.time()
            boxes = do_detect(m, sized, conf_threshold, nms_threshold, use_cuda)
            finish = time.time()

            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))


        result_image_path = '/media/amafla/ssd/pytorch-yolo2-master/retrieval_results/'+imgfile
        # plot_boxes(img, boxes, words, neigh, result_image_path, class_names)
        write_retrieval(img, boxes, result_image_path, class_names)

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


if __name__ == '__main__':

    imgfolder = '/media/amafla/ssd/pytorch-yolo2-master/data/ret_test/'
    #imgfolder = '/home/amafla/Documents/Datasets/IC13/test/'

    cfgfile = 'cfg/yolo-recognition-13anchors.cfg'
    weightfile = 'backup/000041.weights'
    detect(cfgfile, weightfile, imgfolder)
    print ("OPERATION COMPLETE..!!")
    '''
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        detect(cfgfile, weightfile, imgfile)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
    '''
