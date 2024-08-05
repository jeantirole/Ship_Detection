import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
from mmdet.apis import init_detector
from mmrotate.apis import inference_detector_by_patches
import mmrotate
import torch
import cv2
import numpy as np
from mmengine.config import Config
from easydict import EasyDict


#------------- GET Model Inference --------------------------------------------------#

def infer_model(input_file_path,checkpoint_file, config_file,thres,gpu_id):
	
	img = cv2.imread(input_file_path)
	#-----
	cfg = Config.fromfile(config_file)

	#----- classes
	cfg.model.roi_head.bbox_head[0].num_classes=1 
	cfg.model.roi_head.bbox_head[1].num_classes=1

	classes = ('ship',)	
	
	cfg.data.train.classes=classes
	cfg.data.val.classes=classes
	cfg.data.test.classes=classes

	cfg.gpu_ids = gpu_id
	cfg.device='cuda'
	cfg.seed=22

	model = init_detector( config_file, checkpoint_file, device=f'cuda:{cfg.gpu_ids}')
	print("#------------------- init detection")
	
	args = EasyDict()
	args.img = img

	#---------------------------------------------
	args.patch_sizes = [1024]
	args.patch_steps = [896]
	args.img_ratios = [1.0]
	args.merge_iou_thr = 0.1
	args.infer_batch_size= 32

	'''
	#1 차중 위성영상 parmas
	args.patch_sizes = [1024]
	args.patch_steps = [896]
	args.img_ratios = [1.0]
	args.merge_iou_thr = 0.1
	args.infer_batch_size= 32
 
	#2 Sentinel params
	args.patch_sizes = [512] # 512
	args.patch_steps = [384] # 384
	args.img_ratios = [0.5,1.0,1.5]
	args.merge_iou_thr = 0.1
	args.infer_batch_size= 64
	
	'''
	#---------------------------------------------

	args.score_thr = thres
	args.palette = 'dota'

	result = inference_detector_by_patches( model, 
                                            args.img, 
                                            args.patch_sizes,
											args.patch_steps, 
                                        	args.img_ratios,
											args.merge_iou_thr,
											args.infer_batch_size
                                           )

	#output_file_name = [i for i in whole_png.split("/")][-1] + "_result"

	return result