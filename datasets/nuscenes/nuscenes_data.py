from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import argparse
import cv2
import os
import json
import detectron2
from pdb import set_trace as bp
from tqdm import tqdm
import numpy as np

dataroot = "/usr0/tma1/datasets/nuscenes"

class NuScenesDataset:
    def __init__(self, args, version='v1.0-trainval'):
        self.dataroot = args.dataroot
        self.nusc = NuScenes(version=version, dataroot=args.dataroot)
        self.file_ids = self._agg_filenames()
        fd = open(os.path.join(os.path.join(args.dataroot, version), args.anno_file))
        self.anno_dict = json.load(fd)
        fd = open(os.path.join(os.path.join(args.dataroot, version), args.category_file))
        categories = json.load(fd)
        self.category_dict = {v['name']: i for i, v in enumerate(categories)}

def agg_filenames():
    print("Aggregating file names... ")
    file_ids = []
    for i, scene in enumerate(self.nusc.scene):
        token = scene['first_sample_token']
        while (len(token) > 0):
            sample = self.nusc.get('sample', token)
            subsamples = []
            for k, v in sample['data'].items():
                # Only selecting the camera data 'CAM_XXX'
                sample_data = self.nusc.get('sample_data', v)
                if 'CAM' in k and sample_data['is_key_frame']:
                    subsamples.append((sample_data['filename'], sample_data['token']))

            file_ids += subsamples
            token = sample['next']
    print("==> Done!")
    return file_ids

def get_train_data():
    output_file = 'nuscenes_trafficcone_train.json'
    output_path = os.path.join(dataroot, output_file)
    assert os.path.exists(output_path)
    with open(output_path, 'r') as fd:
        data = json.load(fd)
    fd.close()
    data = data
    return data

def get_val_mini_data():
    output_file = 'nuscenes_trafficcone_val.json'
    output_path = os.path.join(dataroot, output_file)
    assert os.path.exists(output_path)
    with open(output_path, 'r') as fd:
        data = json.load(fd)
    fd.close()
    data = data
    return data

def get_val_data():
    output_file = 'nuscenes_trafficcone_val.json'
    output_path = os.path.join(dataroot, output_file)
    assert os.path.exists(output_path)
    print(output_path)
    with open(output_path, 'r') as fd:
        data = json.load(fd)
    fd.close()
    return data

def get_test_data():
    output_file = 'nuscene_test.json'
    output_path = os.path.join(dataroot, output_file)
    assert os.path.exists(output_path)
    with open(output_path, 'r') as fd:
        data = json.load(fd)
    fd.close()
    return data

def get_detectron_data(split):
    output_file = "nuscenes_" + split + '_detectron_2.json'
    output_path = os.path.join(dataroot, output_file)
    assert os.path.exists(output_path)
    with open(output_path, 'r') as fd:
        data = json.load(fd)
    fd.close()
    return data

def scene_id_to_split(splits=['train', 'val', 'test']):
    scene_splits = create_splits_scenes()
    scene_to_split = {}
    for split in splits:
        scenes = scene_splits[split]
        for scene in scenes:
            scene_to_split[scene] = split
    return scene_to_split

def setup_nuscenes(dataroot='/usr0/tma1/datasets/nuscenes'):
    scene_to_split = scene_id_to_split()
    scene_files = {'train': [], 'val': [], 'test': []}

    train_path = os.path.join(dataroot, 'scene_ids_train.json')
    val_path = os.path.join(dataroot, 'scene_ids_val.json')
    test_path = os.path.join(dataroot, 'scene_ids_test.json')
    paths = [train_path, val_path, test_path]

    for version in ['v1.0-trainval', 'v1.0-test']:
        print("Aggregating file names... ")
        file_ids = []
        nusc = NuScenes(version=version, dataroot=dataroot)
        for scene in nusc.scene:
            token = scene['first_sample_token']
            while (len(token) > 0):
                sample = nusc.get('sample', token)
                subsamples = []
                for k, v in sample['data'].items():
                    # Only selecting the camera data 'CAM_XXX'
                    sample_data = nusc.get('sample_data', v)
                    if 'CAM' in k and sample_data['is_key_frame']:
                        subsamples.append((sample_data['filename'], sample_data['token']))

                curr_split = scene_to_split[scene['name']]
                scene_files[curr_split] += subsamples
                token = sample['next']
        print("==> Done!")

    for k, v in scene_files.items():
        print("{}: {}".format(k, len(v)))


    path_keys = ['train', 'val', 'test']
    for i in range(len(paths)):
        with open(paths[i], 'w+') as fd:
            json.dump(scene_files[path_keys[i]], fd)
        fd.close()
    return scene_files

def build_detectron_data(dataroot, file_ids, anno_dict=None, category_dict=None):
    data_list = []
    for file_name, file_id in tqdm(file_ids):
        data = dict()
        data['file_name'] = os.path.join(dataroot, file_name) # To remove new line
        im = cv2.imread(os.path.join(dataroot, file_name))
        if np.any(im) == None: # photo may be damaged
            continue

        data['height'], data['width'] = im.shape[0], im.shape[1]
        data['image_id'] = file_id

        # Do not include instances without annotation
        if anno_dict != None:
            has_anno = anno_dict.get(file_name, None) != None
            if not has_anno:
                continue
            
            # TO-DO: Load data annotation file information
            annotations = []
            for anno_data in anno_dict[file_name]:
                annotation = {}
                annotation['bbox'] = anno_data['bbox_corners']
                annotation['bbox_mode'] = 0 # detectron2.structures.BoxMode.XYXY_ABS
                annotation['category_id'] = category_dict[anno_data['category_name']]
                annotations.append(annotation)
            data['annotations'] = annotations
        data_list.append(data)
    return data_list

if __name__ == '__main__':
    dataroot = '/usr0/tma1/datasets/nuscenes'
    anno_file = 'image_annotations_grouped.json'
    category_file = 'category.json'
    version = 'v1.0-trainval'
    fd = open(os.path.join(os.path.join(dataroot, version), anno_file))
    anno_dict = json.load(fd)
    fd = open(os.path.join(os.path.join(dataroot, version), category_file))
    categories = json.load(fd)
    category_dict = {v['name']: i for i, v in enumerate(categories)}

    # scene_files = setup_nuscenes()
    path_keys = ['train', 'val', 'test']
    train_path = os.path.join(dataroot, 'scene_ids_train.json')
    val_path = os.path.join(dataroot, 'scene_ids_val.json')
    test_path = os.path.join(dataroot, 'scene_ids_test.json')
    paths = [train_path, val_path, test_path]
    scene_files = {'train': [], 'val': [], 'test': []}

    for i in range(len(path_keys)):
        if os.path.exists(paths[i]):
            with open(paths[i], 'r') as fd:
                scene_files[path_keys[i]] = json.load(fd)
    
    print("Building Detectron data...")
    for split, scene_ids in scene_files.items():
        print("for {}".format(split))
        if 'test' not in split:
            d = build_detectron_data(dataroot, scene_ids, anno_dict, category_dict)
        else:
            d = build_detectron_data(dataroot, scene_ids)

        filename = "nuscene_{}.json".format(split)
        print("Saving file to {}...".format(filename))

        with open(os.path.join(dataroot, filename), 'w+') as fd:
            json.dump(d, fd)

