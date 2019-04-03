import sys
import time
import os
from PIL import Image, ImageDraw, ImageFile
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
from tqdm import tqdm
from shutil import move

def detect(cfgfile, weightfile, imgfolder, destination_folder):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    namesfile = 'data/recognition.names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    class_names = load_class_names(namesfile)
    image_list = os.listdir(imgfolder)

    # words, neigh = KNNclassifier()
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    for imgfile in tqdm(image_list):
        img_full_path = imgfolder+imgfile
        #print('Processing image: ', img_full_path)
        img = Image.open(img_full_path).convert('RGB')
        sized = img.resize((m.width, m.height))

        # Paper -> conf_t = 0.0025 and no NMS
        #conf_threshold = 0.0025
        #nms_threshold = 0.0013

        conf_threshold = 0.005
        nms_threshold = 0

        # MOST OF EXPERIMENTS:
        #conf_threshold = 0.025
        #nms_threshold = 0.05

        # ORIGINAL?
        # conf_threshold = 0.1
        # nms_threshold = 0.2
        for i in range(1):
            start = time.time()
            # boxes = do_detect(m, sized, conf_threshold, nms_threshold, use_cuda)
            # TO CHECK BOXES
            boxes = do_detect_retrieval(m, sized, conf_threshold, nms_threshold, use_cuda)
            finish = time.time()

            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        # result_image_path = '/SSD/pytorch-yolo2-master/results/iiit_results/' + imgfile
        # result_image_path = '/SSD/pytorch-yolo2-master/results/sports10k/' + imgfile
        # result_image_path = '/SSD/pytorch-yolo2-master/results/svt1/' + imgfile

        # FINE GRAINED CLASSIFICATION
        result_image_path = destination_folder + imgfile

        write_retrieval(img, boxes, result_image_path, class_names)
        #processed_path = '/home/amafla/Documents/Product_Dataset/processed_images/'
        #move_image(img_full_path, processed_path)

        # PHOC RESULTS FROM TRAINING DATA
        #result_image_path = '/SSD/LG_modified_synth/yolo_phoc_results/synth/' + imgfile


        # ONLY FOR DRINK BOTTLE DATASET
        # result_image_path = destination_folder+imgfile

        # plot_boxes(img, boxes, words, neigh, result_image_path, class_names)
        # write_retrieval(img, boxes, result_image_path, class_names)


if __name__ == '__main__':


    #imgfolder = '/SSD/Datasets/IIIT_STR_V1.0/imgDatabase/'
    #imgfolder = '/SSD/Datasets/sports10K/imgDatabase/'
    #imgfolder = '/SSD/Datasets/svt1/img/'
    #imgfolder = '/home/amafla/Documents/Datasets/IC13/test/'

    # FINE GRAINED CLASSIFICATION
    imgfolder = '/SSD/VQA/TestRepository/VisualGenome/2/'
    destination_folder = '/SSD/VQA/TestRepository/Results_Raw_Phocs/VisualGenome/2/'
    # TO OBTAIN PHOCS FROM TRAINING DATA

    #imgfolder = '/SSD/LG_modified_synth/SynthText_90KDict/images/'

    cfgfile = 'cfg/yolo-recognition-13anchors.cfg'
    weightfile = 'backup/000041.weights'

    detect(cfgfile, weightfile, imgfolder, destination_folder)

    '''For Bottle Dataset
    for i in range(1, 21):
        imgfolder = '/SSD/Datasets/drink_dataset/images/' + str(i) + '/'
        destination_folder = '/home/amafla/yolophoc_bottles/' + str(i) + '/'
        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)
        detect(cfgfile, weightfile, imgfolder, destination_folder)

    '''

    print ("OPERATION COMPLETE..!!")

