import cPickle
from utils import *
from darknet import Darknet
import os
from tqdm import tqdm

# Detect if there is text present in an image and store it

def detect(cfgfile, weightfile, imgfolder, text_file):
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

    #words, neigh = KNNclassifier()
    total_seen = 0
    total_text = 0

    with open(text_file, 'a') as f:

        for imgfile in tqdm(image_list):
            total_seen += 1
            img_full_path = imgfolder + imgfile
            try:
                img = Image.open(img_full_path).convert('RGB')
            except IOError:
                #os.remove(img_full_path)
                print("Error Opening: %s ... Continuing next image" %(img_full_path))
                continue

            sized = img.resize((m.width, m.height))

            conf_threshold = 0.9
            # ORIGINAL
            nms_threshold = 0.8

            # TEST  with better results
            # nms_threshold = 0.2

            for i in range(1):
                start = time.time()
                boxes = do_detect(m, sized, conf_threshold, nms_threshold, use_cuda)
                finish = time.time()
                #print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

            # RESULT PATH
            #result_image_path = destination_folder + imgfile

            if len(boxes) >= 1:
                total_text += 1
                # print("Text found on %s .... Images: %d/%d " % (imgfile, total_text, total_seen))
                f.write(str(imgfile)+'\n')

            #os.remove(img_full_path)
    f.close()


        # Copy images if there is text:
        # count_boxes(img_full_path, boxes, result_image_path)




if __name__ == '__main__':

    imgfolder = '/SSD/new_Product_Dataset/final_images/'
    text_file = '/SSD/new_Product_Dataset/Images_with_Text.txt'

    #if not os.path.exists(destination_folder):
    #   os.mkdir(destination_folder)

    cfgfile = 'cfg/yolo-recognition-13anchors.cfg'
    weightfile = 'backup/000041.weights'

    detect(cfgfile, weightfile, imgfolder, text_file)
    print ("OPERATION COMPLETE..!!")

