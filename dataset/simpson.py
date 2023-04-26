import math
import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry
import random
import argparse

'''
    Included an example.simpson.ipynb for reference (ran on colab)
'''

# Default configs
CLASS_LIMIT = 99 # simpson dataset have 18 classes
VALIDATION_SIZE = 0.5 # minium validation size for 19 classed to be validly trained
BASE_DIR = './data/simpson'
IMAGE_SUBFOLDER = 'simpsons_dataset'

'''
    ! IMPORTANT TRAINING NOTE:
        FOLLOW THESE GUIDELINES IF YOU LIKE THE MODEL TO MAKE ANY PROPER PREDICTIONS AT ALL.
        -- these are from my own experiment with the model and simpson dataset (very large with lots of classes) --
    1. Data size is very important: the more classes are trained, the MORE DATA NEEDED
        Successfully train on: (meaning at least needs)
        - 2 classes -> 0.02 train size
        - 5 classes -> 0.05 train size
        - 10 classes -> 0.2 train size
        - 18 (all) classes -> 0.5 train size
    2. Training should at least pass through the whole training set ONCE.
        So pay attention to `TRAIN.LR_SCHEDULE` and tweak accordingly
        And try to make sure that (in the logged output) -> "Total passes of the training set is:" >= 1
    3. If your training metrics starts to show results as NAN -> Likely failed and will give no predictions
        Ex: total_cost: 0.34688
            wd_cost: 0.1622
        params should be numbers like so, not nan
    4. Epoch size `TRAIN.STEPS_PER_EPOCH` is not that important
        But model saving and other callbacks will be called periodically based on epoch, 
        so smaller value means more checkpoint saving and takes more time
    5. Conclusion:
        - Have enough data
        - Raise TRAIN.LR_SCHEDULE to pass though the training set at least once
        - Train with proper TRAIN.STEPS_PER_EPOCH and see results
'''

class SimpsonDemo(DatasetSplit):
    def __init__(self, base_dir, split, image_subfolder=IMAGE_SUBFOLDER):
        assert split in ["train", "val"]
        base_dir = os.path.expanduser(base_dir)
        self.imgdir = os.path.join(base_dir, image_subfolder)
        self.base_dir = base_dir
        self.split = split
        assert os.path.isdir(self.imgdir), self.imgdir

    '''
        training_roidbs just reads from annotation.json and parse to correct format
    '''
    def training_roidbs(self):
        json_file = os.path.join(self.base_dir, "annotations.json")
        with open(json_file) as f:
            obj = json.load(f)[self.split]
        
        formated = map(lambda val: {
            'file_name': val['file_name'],
            'boxes' : np.asarray(val['boxes'], dtype=np.float32),
            'class' : np.asarray(val['class'], dtype=np.int32),
            'is_crowd': np.asarray( [0], dtype=np.int8),
        }, obj)
        
        result = list(formated)

        print('Example roidb: ', result[0])
        ''' Should be:
            {
                'file_name': './data/balloon/train/34020010494_e5cb88e1c4_k.jpg', 
                'boxes': array([[ 994.5,  619.5, 1445.5, 1166.5]], dtype=float32), 
                'class': array([1], dtype=int32), 
                'is_crowd': array([0], dtype=int8)
            }
        '''
        return result



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
        validation_size,
        class_limit,
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

def register_simpson(basedir):
    try:
        json_file = os.path.join(basedir, "annotations.json")
        with open(json_file) as f:
            obj = json.load(f)
        class_names = list(obj['classes']['mapping'].keys())
        for split in ['train', 'val']:
            name = "simpson_" + split
            DatasetRegistry.register(name, lambda x=split: SimpsonDemo(basedir, x))
            DatasetRegistry.register_metadata(name, "class_names", class_names)
    except Exception as e:
        print('WRN: If your not training/testing on simpson dataset, ignore this error!')
        print(e)
        print('----')

def test_data_visuals(basedir, visualize_subfolder = 'temp_output'):
    from PIL import Image
    from viz import draw_annotation
    import cv2
    from config import config as cfg


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
    parser.add_argument()
    args = parser.parse_args()
    
    # Main part
    process_annotations(args.basedir, args.class_limit, args.validation_size)

    if args.visualize:
        test_data_visuals(args.basedir)
