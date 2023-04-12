import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry
import pandas as pd
import random
import pprint
from config import config as cfg # TODO: remove later

# Stolen link: https://github.com/duckrabbits/ObjectDetection/blob/master/model/parser.py
__all__ = ["register_simpson"]


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
            obj = json.load(f)

        return obj[self.split]




def process_annotations(basedir, image_subfolder = 'simpsons_dataset', validation_size = 0.2):
    # write the final annotation to two separate files
    annotation_file = os.path.join(basedir, "annotation.txt")
    classes_count = {}
    class_mapping = {}
    all_imgs = {}

    with open(annotation_file,'r') as f:
        print('Parsing annotation files')
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split

            # Remove leading '/character' in filename
            filename = os.path.join(basedir, image_subfolder + filename[12:])
            # TODO: file path is wrong, fix it.
            # update class count
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping) + 1
                # this means first class is 1, second is 2, ...

            if filename not in all_imgs:
                all_imgs[filename] = {}
                all_imgs[filename]['file_name'] = filename
                all_imgs[filename]['boxes'] = [[
                    int(val) for val in [x1,y1, x2, y2]
                ]]
                all_imgs[filename]['class'] = [class_mapping[class_name]]

                # determine train or validation set 
                validation_threshold = int(validation_size * 100)
                if np.random.randint(0, 100) > validation_threshold:
                    all_imgs[filename]['imageset'] = 'train'
                else:
                    all_imgs[filename]['imageset'] = 'val'

        train_meta = []
        val_meta = []
        for key in all_imgs:
            if all_imgs[key]['imageset'] == 'train':
                train_meta.append(all_imgs[key])
            else:
                val_meta.append(all_imgs[key])

        classes_count['BG'] = 0
        class_mapping['BG'] = len(class_mapping) + 1
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

def register_simpson(basedir):
    process_annotations(basedir)
    json_file = os.path.join(basedir, "annotations.json")
    with open(json_file) as f:
        obj = json.load(f)
    class_names = list(obj['classes']['mapping'].keys())
    for split in ['train', 'val']:
        name = "simpson_" + split
        DatasetRegistry.register(name, lambda x=split: SimpsonDemo(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", class_names)
    pass

if __name__ == '__main__':
    basedir = './data/simpson'
    # process_annotations(basedir)
    cfg.DATA.CLASS_NAMES = ['BG', 'abraham_grampa_simpson', 'apu_nahasapeemapetilon', 'bart_simpson', 'charles_montgomery_burns', 'chief_wiggum', 'comic_book_guy', 'edna_krabappel', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lisa_simpson', 'marge_simpson', 'milhouse_van_houten', 'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'principal_skinner', 'sideshow_bob']
    roidbs = SimpsonDemo(basedir, "train").training_roidbs()
    print("#images:", len(roidbs))

    from viz import draw_annotation
    # from tensorpack.utils.viz import interactive_imshow as imshow
    from google.colab.patches import cv2_imshow

    import cv2
    for r in roidbs:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"])
        cv2_imshow(vis)
