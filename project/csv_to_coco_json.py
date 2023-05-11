#!/usr/bin/env python3

import itertools
import pandas as pd
import json
from shapely.geometry import Polygon
from shapely.validation import *
from collections import  defaultdict
from iteration_utilities import unique_everseen
from argparse import ArgumentParser

def file_idx_map(filename_list):
    d_dict = defaultdict(lambda: len(d_dict))
    list_ids= [d_dict[n] for n in filename_list]
    file_id_map = defaultdict()
    for filename, idx in zip(filename_list, list_ids):
        file_id_map[filename] = idx+1
    return file_id_map

def CSV2JSON(args):
    csv_path = args.csv
    object_class_map = {'1': 1}
    data = pd.read_csv(csv_path)
    filename_list = []

    image_dict = {}
    image_dict_list = []

    ann_dict = {}
    ann_dict_list = []

    instance_id = 0

    for idx, row in data.iterrows():
        
        points = row.values[5]
        points_dict = json.loads(points)
        x_list = points_dict['all_points_x']
        y_list = points_dict['all_points_y']
        xy_list = []

        for x,y in zip(x_list, y_list):
            xy_tuple = (x,y)
            xy_list.append(xy_tuple)

        polygon = Polygon(xy_list)
        polygon = polygon.buffer(0)
        
        filename = row.values[0]
        print(filename)
        filename_list.append(filename)
        file_id_map = file_idx_map(filename_list)
        image_id = file_id_map[filename]
        
        meta_data = row.values[6]
        print("***")
        print(meta_data)
        # breakpoint()
        meta_data_dict = json.loads(meta_data)
        
        object_class = meta_data_dict['ball']
        object_class_id = object_class_map[object_class]
        
        if polygon.is_valid:
            x0, y0, x1, y1 = polygon.bounds
            bbox = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
            bbox_area = (x1 - x0) * (y1 - y0)
            
            if polygon.geom_type == 'MultiPolygon':
                print("Fix Multipolygon: ", filename, instance_id, object_class, row)
            
            ann_dict["id"] = instance_id
            ann_dict["image_id"] = image_id
            ann_dict["object_classegory_id"] = object_class_id
            ann_dict["segmentation"] = [list(itertools.chain(*list(polygon.exterior.coords)))]
            ann_dict["bbox"] = bbox
            ann_dict["area"] = bbox_area
            ann_dict["iscrowd"] = 0
            ann_dict_list.append(ann_dict)
            ann_dict = {}
            instance_id += 1

        # breakpoint()
        image_dict["id"] = image_id
        image_dict["file_name"] = filename
        image_dict["width"] = 640
        image_dict["height"] = 480
        image_dict["date_captured"] = ""

        image_dict_list.append(image_dict)
        image_dict = {}

    image_dict_list = list(unique_everseen(image_dict_list))

    object_class_dict = {}
    object_class_dict_list = []
    for name, idx in object_class_map.items():
        object_class_dict["superobject_classegory"] = "ballTracker"
        object_class_dict["id"] = int(idx)
        object_class_dict["name"] = name
        object_class_dict_list.append(object_class_dict)

        object_class_dict = {}

    COCO_JSON_dict = {}
    COCO_JSON_dict["images"] = image_dict_list
    COCO_JSON_dict["annotations"] = ann_dict_list
    COCO_JSON_dict["object_classegories"] = object_class_dict_list

    JSON_filename = args.json
    with open(JSON_filename, "w") as f:
        json.dump(COCO_JSON_dict, f)
    print("json saved")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--csv', type=str, default='')
    parser.add_argument(
        '--json', type=str, default='')
    args = parser.parse_args()
    CSV2JSON(args)

