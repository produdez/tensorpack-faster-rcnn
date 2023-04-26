import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry
from parse_simpson import IMAGE_SUBFOLDER

'''
    Included an example.simpson.ipynb for reference (ran on colab)
'''


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
