import math
import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry
import pandas as pd
import random
import pprint
from config import config as cfg # TODO: remove later

# Stolen link: https://github.com/duckrabbits/ObjectDetection/blob/master/model/parser.py

# CLASS_LIMIT = 10
CLASS_LIMIT = 99

'''
    ! IMPORTANT TRAINING NOTE:
        FOLLOW THESE GUIDELINES IF YOU LIKE THE MODEL TO MAKE ANY PROPER PREDICTIONS AT ALL.
        -- these are from my own experiment with the model and simpson dataset (very large with lots of classes) --
    1. Data size is very important: the more classes are trained, the MORE DATA NEEDED
        Successful train on:
        - 2 classes -> 0.02 train size
        - 5 classes -> 0.05 train size
        - 10 classes -> 0.2 train size
        - 18 (all) classes -> 0.8 train size
    2. Training should at least pass through the whole training set ONCE.
        So pay attention to `TRAIN.LR_SCHEDULE`
        And try to make sure that (in the logged output) -> "Total passes of the training set is:" >= 1
    3. If your training metrics starts to show results as NAN -> Likely failed and will give no predictions
        Ex: total_cost: 0.34688
            wd_cost: 0.1622
        params should be numbers like so, not nan
    4. Epoch size `TRAIN.STEPS_PER_EPOCH` is not that important
        But model saving and other callbacks will be called periodically based on epoch, 
        so smaller value means more checkpoint saving and takes more time
    
'''
class SimpsonDemo(DatasetSplit):
    def __init__(self, base_dir, split, image_subfolder = 'simpsons_dataset'):
        assert split in ["train", "val"]
        base_dir = os.path.expanduser(base_dir)
        self.imgdir = os.path.join(base_dir, image_subfolder)
        self.base_dir = base_dir
        self.split = split
        assert os.path.isdir(self.imgdir), self.imgdir


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
        # {'file_name': './data/balloon/train/34020010494_e5cb88e1c4_k.jpg', 'boxes': array([[ 994.5,  619.5, 1445.5, 1166.5]], dtype=float32), 'class': array([1], dtype=int32), 'is_crowd': array([0], dtype=int8)}
        return result



# def process_annotations(basedir, image_subfolder = 'simpsons_dataset', validation_size = 0.7):
def process_annotations(basedir, image_subfolder = 'simpsons_dataset', validation_size = 0.5):
    # write the final annotation to two separate files
    annotation_file = os.path.join(basedir, "annotation.txt")
    classes_count = {}
    class_mapping = {}
    all_imgs = {}

    # NOTE: if you want to remove this, then you must also change how classes are added to frame work
    classes_count['BG'] = 0
    class_mapping['BG'] = 0

    with open(annotation_file,'r') as f:
        print('Parsing annotation files')
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split
            x1, y1, x2, y2 = [int(x) + 0.5 for x in [x1, y1, x2, y2]]

            x1, x2 = min(x1,x2), max(x1,x2)
            y1, y2 = min(y1,y2), max(y1,y2)
            # !MAKE SURE BOUNDING BOX IS VALID
            area = (x2 - x1) * (y2 - y1)

            if area <= 0: continue
            
            #! Remove leading '/character' or '/character2' in filename
            filename = '/'.join(filename.split('/')[2:])
            filename = os.path.join(basedir, image_subfolder, filename) 



            if class_name not in class_mapping:
                if len(class_mapping) > CLASS_LIMIT: continue
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


        # NOTE: Train size will try to have balance of classes and also no class is missing from training set
        min_size,  _ = max(zip(classes_count.values(), classes_count.keys()))
        train_threshold = math.ceil((1 - validation_size) * min_size)
        training_counter = {key : train_threshold for key in classes_count.keys()}
        train_meta = []
        val_meta = []
        
        image_keys = list(all_imgs.keys())
        random.shuffle(image_keys)
        for key in image_keys:
            classname = all_imgs[key]['classname'][0]
            if training_counter[classname] >= 0:
                train_meta.append(all_imgs[key])
                training_counter[classname] -= 1
            else:
                val_meta.append(all_imgs[key])

        print('Training images per class ({} classes) :'.format(len(classes_count)))
        pprint.pprint(classes_count)
        print(f'Train size: {len(train_meta)}, validation size: {len(val_meta)}')

    # write to files
    with open(os.path.join(basedir, 'annotations.json'), 'w') as fw:
        json.dump({
            'classes' : {
                'count' : classes_count,
                'mapping' : class_mapping
            },
            'train': train_meta, 
            'val': val_meta,
        }, fw, indent=4)

def register_simpson(basedir, process_raw_annotations=True):
    # TODO: fix major bug cause if you process annotations here then you cant register other data sets to train them!!!!
    if basedir.split('/')[-1] != 'simpson': return
    if process_raw_annotations: process_annotations(basedir) 
    json_file = os.path.join(basedir, "annotations.json")
    with open(json_file) as f:
        obj = json.load(f)
    class_names = list(obj['classes']['mapping'].keys())
    for split in ['train', 'val']:
        name = "simpson_" + split
        DatasetRegistry.register(name, lambda x=split: SimpsonDemo(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", class_names)

def test_data_visuals():
    basedir = './data/simpson'

    process_annotations(basedir)
    json_file = os.path.join(basedir, "annotations.json")
    with open(json_file) as f:
        obj = json.load(f)
    class_names = list(obj['classes']['mapping'].keys())
    cfg.DATA.CLASS_NAMES = class_names
    
    roidbs = SimpsonDemo(basedir, "train").training_roidbs()
    print("#images:", len(roidbs))

    from PIL import Image
    from viz import draw_annotation
    import cv2

    visualization_folder = os.path.join(basedir, 'temp_output')
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
    test_data_visuals()
