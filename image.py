#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np
import logging
import re

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    

    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)
    # WE DONT WANT TO FLIP TEXT
    # flip = random.randint(1,10000)%2
    # if flip:
    #   sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    flip = 0
    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 100
    label = np.zeros((max_boxes,5))
    word_list = list()
    if os.path.getsize(labpath):

        text_file = open(labpath, 'r')
        lines = text_file.read().split('\n')
        ordered_lines = list ()
        for i in range (0, len(lines)):
            if lines[i] == '':
                break
            temporal_list = lines[i].split(" ")
            word_list.append(temporal_list[0])
            temporal_list [0] = 0
            ordered_lines.append(temporal_list)

        bs = np.asarray(ordered_lines, dtype=np.float32)
        if bs is None:
            return label, word_list
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]): # ITERATE THROUGH EVERY GROUND TRUTH OBJECT
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            # IF OBJECT IS TOO SMALL/NOISE DISCARD
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= max_boxes:
                break

    label = np.reshape(label, (-1))
    return label, word_list

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label, word_list = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
    # print (word_list)
    phoc_label = fill_phoc(word_list)

    return img,label, phoc_label



def fill_phoc(word_list):
    max_boxes = 100
    phoc_matrix = np.zeros((max_boxes, 604))
    if len(word_list) == 0:
        return phoc_matrix
    for i in range(0, len(word_list)):
        phoc_matrix[i,:] = phoc(word_list[i])
    return phoc_matrix

def build_phoc(words, phoc_unigrams, unigram_levels,
               bigram_levels=None, phoc_bigrams=None,
               split_character=None, on_unknown_unigram='error'):
    '''
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).
    Args:
        word (str): word to calculate descriptor for
        phoc_unigrams (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels for the unigrams in PHOC
        phoc_bigrams (list of str): list of bigrams to be used in the PHOC
        phoc_bigram_levls (list of int): the levels of the bigrams in the PHOC
        split_character (str): special character to split the word strings into characters
        on_unknown_unigram (str): What to do if a unigram appearing in a word
            is not among the supplied phoc_unigrams. Possible: 'warn', 'error'
    Returns:
        the PHOC for the given word
    '''
    # prepare output matrix
    logger = logging.getLogger('PHOCGenerator')
    if on_unknown_unigram not in ['error', 'warn']:
        raise ValueError('I don\'t know the on_unknown_unigram parameter \'%s\'' % on_unknown_unigram)
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
    if phoc_bigrams is not None:
        phoc_size += len(phoc_bigrams) * np.sum(bigram_levels)
    phocs = np.zeros((len(words), phoc_size))
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    # iterate through all the words
    for word_index, word in enumerate(words):
        if split_character is not None:
            word = word.split(split_character)
        n = len(word)
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            if char not in char_indices:

                if on_unknown_unigram == 'warn':
                    logger.warn('The unigram \'%s\' is unknown, skipping this character', char)
                    continue
                else:
                    logger.fatal('The unigram \'%s\' is unknown', char)
                    print (char)
                    #raise ValueError()
            char_index = char_indices[char]
            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(
                            phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        phocs[word_index, feat_vec_index] = 1

                # add bigrams
        if phoc_bigrams is not None:
            ngram_features = np.zeros(len(phoc_bigrams) * np.sum(bigram_levels))
            ngram_occupancy = lambda k, n: [float(k) / n, float(k + 2) / n]
            for i in range(n - 1):
                ngram = word[i:i + 2]
                phoc_dict = {k: v for v, k in enumerate(phoc_bigrams)}
                if phoc_dict.get(ngram, 666) == 666:
                    continue
                occ = ngram_occupancy(i, n)
                for level in bigram_levels:
                    for region in range(level):
                        region_occ = occupancy(region, level)
                        overlap_size = size(overlap(occ, region_occ)) / size(occ)
                        if overlap_size >= 0.5:
                            ngram_features[region * len(phoc_bigrams) + phoc_dict[ngram]] = 1
            phocs[word_index, -ngram_features.shape[0]:] = ngram_features

    return phocs

def phoc(raw_word):
    '''

    :param raw_word: string of word to be converted
    :return: phoc representation as a np.array (1,604)
    '''

    word =[raw_word]
    word_lowercase = word[0].lower()
    word = [word_lowercase]
    phoc_unigrams = 'abcdefghijklmnopqrstuvwxyz0123456789'
    unigram_levels = [2,3,4,5]
    bigram_levels=[]
    bigram_levels.append(2)

    phoc_bigrams = []
    i = 0
    with open('/SSD/pytorch-yolo2-master/PHOC/bigrams_new.txt','r') as f:
        for line in f:
            a = line.split()
            phoc_bigrams.append(a[0].lower())
            #phoc_bigrams.append(list(a[0])[0])
            #phoc_bigrams.append(list(a[0])[1])
            i = i +1
            if i >= 50:break


    qry_phocs = build_phoc(words = word, phoc_unigrams = phoc_unigrams, unigram_levels = unigram_levels,
                           bigram_levels = bigram_levels, phoc_bigrams = phoc_bigrams)

    return qry_phocs
