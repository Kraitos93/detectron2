from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
import sys
from detectron2.data.datasets import bottle_loader
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def gen_cfg_train(model, weights, dataset):
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-Detection/" + model)
    cfg.DATASETS.TRAIN = (dataset + '_train',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/" + os.path.splitext(model)[0] + '/' + weights  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon)
    cfg.OUTPUT_DIR = 'output_' + dataset
    return cfg

def gen_cfg_test(dataset):
    #cfg = gen_cfg_train(model, weights, dataset)
    cfg = get_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = (dataset + '_test', )
    return cfg

def train_model(path, model, weights, dataset):
    bottle_loader.register_dataset(path, dataset)
    cfg = gen_cfg_train(model, weights, dataset)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

def test_model(path, model, weights, dataset):
    bottle_loader.register_dataset(path, dataset)
    cfg_test = gen_cfg_test(dataset)
    cfg = gen_cfg_train(model, weights, dataset)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    evaluator = COCOEvaluator("%s_test" % (dataset), cfg_test, False, output_dir="./output_%s/" % (dataset))
    val_loader = build_detection_test_loader(cfg_test, "%s_test" % (dataset))
    inference_on_dataset(trainer.model, val_loader, evaluator)

def main(args):
    run = args[1]
    path = args[2]
    if run == 'train':
        train_model(args[2], args[3], args[4], args[5])
    elif run == 'test':
        test_model(args[2], args[3], args[4], args[5])

if __name__ == "__main__":
    main(sys.argv)


#train_model('faster_rcnn_R_50_C4_3x.yaml', '137849393/model_final_f97cb7.pkl', 'bottle')
