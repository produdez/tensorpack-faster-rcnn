# Tensorpack Faster-RCNN on Simpson
This is a moddified version of the original example of F-RCNN from [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).

Which adds the `simpson` dataset to the code.

View the original `README` by going to the tensorpack link above.

## Simpson Tranfer training

- Adding simpson dataset to tensorboard F-RCNN code
- And running example/experiment of transfer learning
- Some moddifications to original code (minor) mostly for easier experimenting

## How to use?
```bash
# after getting code, dependencies, ..., parse the dataset
python -m dataset.parse_simpson --validation-size 0.5
# then train model
python ./train.py --config DATA.BASEDIR=./data/simpson MODE_FPN=True \
  MODE_MASK=False\
	TRAIN.STEPS_PER_EPOCH=1000 \
	"DATA.VAL=('simpson_val',)"  "DATA.TRAIN=('simpson_train',)" \
	TRAIN.BASE_LR=1e-1 TRAIN.EVAL_PERIOD=0 "TRAIN.LR_SCHEDULE=[800]" \
	"PREPROC.TRAIN_SHORT_EDGE_SIZE=[600,1200]" TRAIN.CHECKPOINT_PERIOD=1 DATA.NUM_WORKERS=1 \
	--load COCO-MaskRCNN-R50FPN2x.npz --logdir train_log/simpson

# predict a folder with the trained model
python ./predict.py --config DATA.BASEDIR=./data/simpson MODE_FPN=True MODE_MASK=False \
	"DATA.VAL=('simpson_val',)"  "DATA.TRAIN=('simpson_train',)" \
	--load train_log/simpson/checkpoint --predict-folder {test_destination}
```
Full example in simpson.ipynb
