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

    words, neigh = KNNclassifier()

    for imgfile in image_list:
        img_full_path = imgfolder+imgfile
        img = Image.open(img_full_path).convert('RGB')
        sized = img.resize((m.width, m.height))

        # conf_threshold = 0.35

        conf_threshold = 0.25
        # ORIGINAL
        nms_threshold = 0.5

        # TEST  with better results
        # nms_threshold = 0.2

        for i in range(2):
            start = time.time()
            boxes = do_detect(m, sized, conf_threshold, nms_threshold, use_cuda)
            finish = time.time()

            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        # RESULT PATH
        result_image_path = '/media/amafla/ssd/pytorch-yolo2-master/predictions/' + imgfile
        plot_boxes(img, boxes, words, neigh, result_image_path, class_names)

        # ICDAR 13
        #result_image_path = '/media/amafla/ssd/pytorch-yolo2-master/ic13_txt_results/'+imgfile
        # ICDAR 15
        # result_image_path = '/media/amafla/ssd/pytorch-yolo2-master/ic15_txt_results/'+imgfile

        #write_text_result_icdar(img, boxes, words, neigh, result_image_path, class_names)

if __name__ == '__main__':

    imgfolder = '/media/amafla/ssd/pytorch-yolo2-master/data/test/'
    # imgfolder = '/home/amafla/Documents/Datasets/IC13/test/'
    # imgfolder = '/home/amafla/Documents/Datasets/IC15/test/'
    # imgfolder = '/media/amafla/ssd/pytorch-yolo2-master/overfit_test/'
    cfgfile = 'cfg/yolo-recognition-13anchors.cfg'
    #weightfile = '/media/amafla/ssd/darknet-phoc/backup/yolo-phoc.backup'
    weightfile = 'backup/iam180.weights'

    # weightfile = 'bin/yolo-phoc.weights'
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
