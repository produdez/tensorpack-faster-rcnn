import math
import random
import argparse
from viz import draw_annotation
from PIL import Image
import cv2
import os
from config import config as cfg
from .simpson import SimpsonDemo
import json 
import numpy as np

# Default configs
CLASS_LIMIT = 99 # simpson dataset have 18 classes
VALIDATION_SIZE = 0.5 # minium validation size for 19 classed to be validly trained
BASE_DIR = './data/simpson'
IMAGE_SUBFOLDER = 'simpsons_dataset'


def split_train_val(all_imgs, classes_count, validation_size):
    '''
        This function will try balance of classes
        1. train size is based on the count of the smallest class (class with least samples)
            Meaning if train_size is 0.1 and smallest class have 10 samples then
            -> We take 1 sample from each class of the training set
        2. This ensure a kind-of-balance and uniform distribution
        3. And also no empty training class samples
    '''
    min_size,  _ = max(zip(classes_count.values(), classes_count.keys()))
    train_threshold = math.ceil((1 - validation_size) * min_size)
    training_counter = {key : train_threshold for key in classes_count.keys()}
    train = []
    val = []
    
    image_keys = list(all_imgs.keys())
    random.shuffle(image_keys)
    for key in image_keys:
        classname = all_imgs[key]['classname'][0]
        if training_counter[classname] >= 0:
            train.append(all_imgs[key])
            training_counter[classname] -= 1
        else:
            val.append(all_imgs[key])

    print(f'Classes count: {len(classes_count)}')
    print(f'Train size: {len(train)}, validation size: {len(val)}')
    return train, val

'''
    Parse annotation.txt and write to annotation.json with train/val split for easy
        training without actually splitting the dataset into train/val folder

    NOTE: process_annotations's
        bounding box parsing is based on code from the following project
    https://github.com/duckrabbits/ObjectDetection/blob/master/model/parser.py
'''
def process_annotations(
        basedir, 
        class_limit,
        validation_size,
        image_subfolder=IMAGE_SUBFOLDER
    ):
    print(f'Number of training classes limit: {class_limit}')   
    print(f'Training size: {1 - validation_size}')
    
    annotation_file = os.path.join(basedir, "annotation.txt")
    classes_count = {}
    class_mapping = {}
    all_imgs = {}

    classes_count['BG'] = 0
    class_mapping['BG'] = 0

    with open(annotation_file,'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split
            x1, y1, x2, y2 = [int(x) + 0.5 for x in [x1, y1, x2, y2]]

            x1, x2 = min(x1,x2), max(x1,x2)
            y1, y2 = min(y1,y2), max(y1,y2)
            #! MAKE SURE BOUNDING BOX IS VALID
            area = (x2 - x1) * (y2 - y1)

            if area <= 0: continue
            
            #! Remove leading '/character' or '/character2' in filename
            filename = '/'.join(filename.split('/')[2:])
            filename = os.path.join(basedir, image_subfolder, filename) 



            if class_name not in class_mapping:
                if len(class_mapping) > class_limit: continue
                class_mapping[class_name] = len(class_mapping)
                # this means first class is 1, second is 2, ...
            
            # update class count
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if filename not in all_imgs:
                all_imgs[filename] = {}
                all_imgs[filename]['file_name'] = filename
                all_imgs[filename]['boxes'] = [[x1,y1, x2, y2]]
                all_imgs[filename]['class'] = [class_mapping[class_name]]
                all_imgs[filename]['classname'] = [class_name]

        train_meta, val_meta = split_train_val(all_imgs, classes_count, validation_size)

    # write to file
    with open(os.path.join(basedir, 'annotations.json'), 'w') as fw:
        json.dump({
            'classes' : {
                'count' : classes_count,
                'mapping' : class_mapping
            },
            'train': train_meta, 
            'val': val_meta,
        }, fw, indent=4)


def test_data_visuals(basedir, visualize_subfolder = 'temp_output'):
    json_file = os.path.join(basedir, "annotations.json")
    with open(json_file) as f:
        obj = json.load(f)
    class_names = list(obj['classes']['mapping'].keys())

    #! This is for the draw_annotation function to know our classes
    cfg.DATA.CLASS_NAMES = class_names 
    
    roidbs = SimpsonDemo(basedir, "train").training_roidbs()

    visualization_folder = os.path.join(basedir, visualize_subfolder)
    if not os.path.isdir(visualization_folder): os.mkdir(visualization_folder)

    visual_percentage = 0.01

    for idx, r in enumerate(roidbs):
        im = cv2.imread(r["file_name"])
        
        vis = draw_annotation(im, r["boxes"], r["class"])

        if np.random.randint(0, 100) < (visual_percentage * 100):
            visual_name = f'{idx}.png'
            img = Image.fromarray(vis, 'RGB')
            output_path = os.path.join(visualization_folder, visual_name)
            img.save(output_path)
            print('File: ', r['file_name'], ' visualized as: ', visual_name)

    print('Example data visualizations are in ', visualization_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--basedir', help='Dataset directory',
        type=str, required=False, default=BASE_DIR
    )
    parser.add_argument(
        '--class-limit', help='maximum number of classes to parse and later use in training/prediction',
        type=int, required=False, default=CLASS_LIMIT
    )
    parser.add_argument(
        '--validation-size', help='size of the dataset to use for validation (0 < x < 1)',
        type=float, required=False, default=VALIDATION_SIZE
    )
    parser.add_argument(
        '--visualize', help='If used, small part of the training dataset will be visualized in a subfolder inside the dataset base directory',
        action='store_true'
    )
    args = parser.parse_args()
    
    # Main part
    process_annotations(args.basedir, args.class_limit, args.validation_size)

    if args.visualize:
        test_data_visuals(args.basedir)
