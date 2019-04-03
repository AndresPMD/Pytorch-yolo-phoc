import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

def detect(cfgfile, weightfile, imgfile):
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

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    conf_threshold = 0.35
    nms_threshold = 0.5
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, conf_threshold, nms_threshold, use_cuda)
        finish = time.time()

        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)


if __name__ == '__main__':
    # TRAINING FOR TEXT RECOGNITION on SYNTH:
    #imgfile = '/media/amafla/ssd/LG_modified_synth/SynthText_90KDict_small/images/15592fe254d042c79c592e5f613564a6.jpg'
    imgfile = '/SSD/pytorch-yolo2-master/data/img_1.jpg'
    cfgfile = 'cfg/yolo-recognition-13anchors.cfg'
    weightfile = 'backup/000041.weights'
    detect(cfgfile, weightfile, imgfile)

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
