#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2 import model_zoo
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetMapper
)

from datetime import datetime
import json
from detectron2.utils.logger import setup_logger
setup_logger()

logger = logging.getLogger("detectron2")

def get_data(args, file_name):
    def ret_data():
        output_path = os.path.join(args.dataroot, file_name)
        print(output_path)
        assert os.path.exists(output_path)
        with open(output_path, 'r') as fd:
            data = json.load(fd)
        fd.close()
        return data
    return ret_data

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        dataset_name = cfg.DATASETS.TEST[0]
        evaluator = COCOEvaluator(dataset_name, tasks=["bbox"], output_dir=cfg.OUTPUT_DIR)

        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    print(results)
    return results

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter)

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    now = datetime.now()
    dt_string =  now.strftime("%Y-%m-%d-%H-%M-%S")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Modified configurations
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.TEST.EVAL_PERIOD = 2000
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"


    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = 100000
    cfg.DATASETS.TRAIN = ("nuimages_train_trafficone",)
    cfg.DATASETS.TEST = ("nuimages_val_trafficone",)
    cfg.OUTPUT_DIR = os.path.join('/usr0/tma1/traffic_cone_detection/output', dt_string)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    return cfg

def main(args):
    cfg = setup(args)

    # Register datasets
    category_dict = {0: 'traffic_cones'}
    data_train = DatasetCatalog.register("nuimages_train_trafficone", get_data(args, args.train_file))
    MetadataCatalog.get("nuimages_train_trafficone").thing_classes = category_dict
    data_val = DatasetCatalog.register("nuimages_val_trafficone", get_data(args, args.val_file))
    MetadataCatalog.get("nuimages_val_trafficone").thing_classes = category_dict
    data_test = DatasetCatalog.register("nuimages_test_trafficone", get_data(args, args.test_file))
    MetadataCatalog.get("nuimages_test_trafficone").thing_classes = category_dict

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model).load(args.ckpt_file)
        return do_test(cfg, model)
    
    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--dataroot', type=str, \
                        default='/usr0/tma1/datasets/nuimages/detectron_data')
    parser.add_argument('--train_file', type=str,\
                        default='v1.0-train_trafficcone_detectron.json') 
    parser.add_argument('--val_file', type=str,\
                        default='val_val_trafficcone_detectron.json') 
    parser.add_argument('--test_file', type=str,\
                        default='val_test_trafficcone_detectron.json') 
    parser.add_argument('--ckpt_file', type=str,\
                        default='/usr0/tma1/traffic_cone_detection/output/2021-10-26-01-38-57/model_0076499.pth')

    args = parser.parse_args()

    print("Command Line Args:", args)
    
    main(args)
