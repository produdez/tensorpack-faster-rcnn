import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry
import pandas as pd
import random
import pprint
from config import config as cfg # TODO: remove later

# Stolen link: https://github.com/duckrabbits/ObjectDetection/blob/master/model/parser.py

CLASS_LIMIT = 99

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
        return result



def process_annotations(basedir, image_subfolder = 'simpsons_dataset', validation_size = 0.95):
    # write the final annotation to two separate files
    annotation_file = os.path.join(basedir, "annotation.txt")
    classes_count = {}
    class_mapping = {}
    all_imgs = {}

    # TODO: if you want to remove this, then you must also change how classes are added to frame work
    classes_count['BG'] = 0
    class_mapping['BG'] = 0

    # {'file_name': './data/simpson/simpsons_dataset/abraham_grampa_simpson/pic_0000.jpg', 'boxes': array([[52., 72., 57., 72.]], dtype=float32), 'class': array([1], dtype=int32), 'is_crowd': array([0], dtype=int8)}
    # {'file_name': './data/balloon/train/34020010494_e5cb88e1c4_k.jpg', 'boxes': array([[ 994.5,  619.5, 1445.5, 1166.5]], dtype=float32), 'class': array([1], dtype=int32), 'is_crowd': array([0], dtype=int8)}
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

                # determine train or validation set 
                validation_threshold = int(validation_size * 100) - 1
                if np.random.randint(0, 100) > validation_threshold:
                    all_imgs[filename]['imageset'] = 'train'
                else:
                    all_imgs[filename]['imageset'] = 'val'

        # NOTE: there's a chance that some class is empty in training dataset
        # TODO: make sure all classes are in training dataset
        train_meta = []
        val_meta = []
        for key in all_imgs:
            if all_imgs[key]['imageset'] == 'train':
                train_meta.append(all_imgs[key])
            else:
                val_meta.append(all_imgs[key])

        print('Training images per class ({} classes) :'.format(len(classes_count)))
        pprint.pprint(classes_count)

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
