import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import random

def get_data_dict(dataset_dir):
    images = os.path.join(dataset_dir, 'images')
    images_list = os.listdir(images)
    random.shuffle(images_list)

    train = images_list[0,len(images_list)*0.80]
    test = images_list[len(images_list)*0.80, len(images_list)]

    dataset_dicts_train = []
    dataset_dicts_val = []

    annotation_folder = 'original'
    if len(os.listdir(dataset_dir + '/annotations/transformed')) > 0:
        annotation_folder = 'transformed'
    
    for img in train:
        record = get_data_record(annotation_folder, dataset_dir, img)
        dataset_dicts_train.append(record)
    for img in test:
        record = get_data_record(annotation_folder, dataset_dir, img)
        dataset_dicts_val.append(record)

    return dataset_dicts_train, dataset_dicts_val



def get_data_record(annotation_folder, dataset_dir, img):
    record = {}
    #Get the annotation file for the image
    image_name = os.path.splitext(img)[0]
    with open(os.path.join(dataset_dir, 'annotations', annotation_folder, image_name + '_annotations.json')) as f:
        img_annotation = json.load(f)
    
    filename = os.path.join(dataset_dir, 'images', img)
    height = img_annotation['height']
    width = img_annotation['width']
    idx = img_annotation['idx']
    annotations = img_annotation['annotations']
    objs = []
    for anno in annotations:
        category_name = anno['category']
        bbox = anno['bbox']
        x_top = bbox['x_top']
        x_bootom = bbox['x_bottom']
        y_top = bbox['y_top']
        y_bottom = bbox['y_bottom']
        category_id = category_switch(category_name)
        obj = {
            'bbox': [x_top, y_top, x_bootom, y_bottom],
            'bbox_mode': BoxMode.XYXY_ABS,
            'category_id': category_id
        }
        objs.append(obj)
    record['file_name'] = filename
    record['image_id'] = idx
    record['height'] = height
    record['width'] = width
    record['annotations'] = objs
    return record

def category_switch(category):
    if category == 'pepsi':
        return 0
    elif category == 'mtn_dew':
        return 1
    elif category == 'pepsi_cherry':
        return 2
    elif category == 'pepsi_zerot':
         #TODO: This category is not correct, I think it is zerot or something like that
        return 3
    else:
        raise Exception('Unknown category in the dataset')


from detectron2.data import DatasetCatalog, MetadataCatalog
bottle_train, bottle_test = get_data_dict('soda_bottle')
DatasetCatalog.register('bottle_train', bottle_train)
MetadataCatalog.get('ballon_train').set(thing_classes=['pepsi', 'mtn_dew', 'pepsi_cherry', 'pepsi_zerot'])

DatasetCatalog.register('bottle_test', bottle_test)
MetadataCatalog.get('ballon_test').set(thing_classes=['pepsi', 'mtn_dew', 'pepsi_cherry', 'pepsi_zerot'])

bottle_metadata = MetadataCatalog.get("bottle_train")


