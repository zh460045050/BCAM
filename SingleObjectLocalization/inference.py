"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from cProfile import label
import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd
import torch.nn as nn

from evaluation import BoxEvaluator
from evaluation import MaskEvaluator
from evaluation import configure_metadata
from util import t2n
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


class Generator(object):

    def __init__(self, extractor, candidate_layers="layer4", dataset_name="CUB"):

        self.device = next(extractor.parameters()).device
        self.handlers = []  # a set of hook function handlers
        self.phi = 1
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        self.dataset_name = dataset_name

        def save_fmaps(key):
            def forward_hook(module, input, output):
                if not isinstance(output, dict):
                    self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in extractor.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _encode_one_hot(self, logits, targets):
        #print(self.logits.shape)
        one_hot = torch.zeros_like(logits).to(self.device)
        for i in range(0, one_hot.shape[0]):
            one_hot[i, targets[i]] = 1.0
        return one_hot
        
    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, extractor, image, ids, target_layer, cls_agnostic=False):

        self.image_shape = image.shape[2:]

        out_pict = extractor(image)
        if self.dataset_name == "CUB" or self.dataset_name == "ILSVRC":
            logits = out_pict["logits"] - out_pict["logits_rev"]
        else:
            logits = out_pict["logits"] #- out_pict["logits_rev"]
        extractor.zero_grad()

        self.ids = ids
        one_hot = self._encode_one_hot(logits, ids)
        self.phi = torch.zeros(logits.shape[0], 1).cuda()
        for i in range(0, logits.shape[0]):
            self.phi[i] = logits[i, ids[i]]

        if cls_agnostic:
            one_hot = torch.ones(one_hot.shape).cuda()
        
        logits.backward(gradient=one_hot, retain_graph=True)

        gradient = self._find(self.grad_pool, target_layer)

        weights = F.adaptive_avg_pool2d(gradient, 1)

        if self.dataset_name == "CUB": #or self.dataset_name == "ILSVRC": #or self.dataset_name == "ILSVRC":
            gradient = torch.mul(gradient, weights).sum(dim=1, keepdim=True)
        elif self.dataset_name == "ILSVRC":
            #gradient = self._find(self.grad_pool, target_layer)
            #weights = F.adaptive_avg_pool2d(torch.relu(gradient), 1)
            gradient = torch.mul(gradient, torch.relu(weights)).sum(dim=1, keepdim=True)
            #gradient = torch.relu(gradient)
            '''
            fmaps = self._find(self.fmap_pool, target_layer)
            gradient = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
            '''
        else:
            fmaps = self._find(self.fmap_pool, target_layer)
            gradient = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)

        gradient = F.interpolate(
            gradient, self.image_shape, mode="bilinear", align_corners=False
        ).squeeze(1)

        return out_pict["logits"], gradient


    def generate_back(self, extractor, image, ids, target_layer, cls_agnostic=False):

        self.image_shape = image.shape[2:]

        out_pict = extractor(image)
        if self.dataset_name == "CUB" or self.dataset_name == "ILSVRC":
            logits = out_pict["logits_back"] - out_pict["logits_back_rev"]
        else:
            logits = out_pict["logits_back"] #- out_pict["logits_back_rev"]
            
        extractor.zero_grad()

        self.ids = ids
        one_hot = self._encode_one_hot(logits, ids)
        self.phi = torch.zeros(logits.shape[0], 1).cuda()
        for i in range(0, logits.shape[0]):
            self.phi[i] = logits[i, ids[i]]

        if cls_agnostic:
            one_hot = torch.ones(one_hot.shape).cuda()

        logits.backward(gradient=one_hot, retain_graph=True)

        gradient = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(gradient, 1)

        if self.dataset_name == "CUB": #or self.dataset_name == "ILSVRC":
            gradient = torch.mul(gradient, weights).sum(dim=1, keepdim=True)
        elif self.dataset_name == "ILSVRC":
            gradient = torch.mul(gradient, torch.relu(weights)).sum(dim=1, keepdim=True)
            #gradient = torch.relu(gradient)
            '''
            fmaps = self._find(self.fmap_pool, target_layer)
            gradient = torch.relu(gradient)
            weights = F.adaptive_avg_pool2d(gradient, 1)
            gradient = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
            '''
        else:
            fmaps = self._find(self.fmap_pool, target_layer)
            gradient = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)

        gradient = F.interpolate(
            gradient, self.image_shape, mode="bilinear", align_corners=False
        ).squeeze(1)

        return out_pict["logits"], gradient

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None, wsol_method='cam', target_layer='layer4', is_vis=False, eval_type='prob'):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        #
        self.wsol_method = wsol_method
        #
        self.target_layer = target_layer

        self.eval_type = eval_type
        self.is_vis = is_vis

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.dataset_name = dataset_name

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_cams(self):

        if self.wsol_method == 'bcam':
            gcam = Generator(self.model, candidate_layers=self.target_layer, dataset_name=self.dataset_name)

        print("Computing and evaluating cams.")
        count = 0
        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            images = images.cuda()
            ####
            count = count + 1
            if self.wsol_method == 'bcam': 

                if "cls_agnostic" in self.eval_type:
                    logits, cams = gcam.generate(self.model, images, targets, target_layer=self.target_layer, cls_agnostic=True)
                    _, cams_back = gcam.generate_back(self.model, images, targets, target_layer=self.target_layer, cls_agnostic=True)
                else:
                    logits, cams = gcam.generate(self.model, images, targets, target_layer=self.target_layer)
                    _, cams_back = gcam.generate_back(self.model, images, targets, target_layer=self.target_layer)

                predicts = torch.argmax(logits, dim=1)
                predicts = t2n(predicts.view(predicts.shape[0], 1))
                targets = t2n(targets)

                cams = t2n(cams)
                cams_back = t2n(cams_back)

                for cam, cam_back, image, image_id, predict, target in zip(cams, cams_back, images, image_ids, predicts, targets):
                    
                    #print(cam.shape)
                    cam_resized = cv2.resize(cam, image_size,
                                            interpolation=cv2.INTER_CUBIC)# * att_resized

                    fore = torch.from_numpy(cam_resized.copy()).unsqueeze(0)


                    cam_normalized = normalize_scoremap(cam_resized)

                    cam_resized_back = cv2.resize(cam_back, image_size,
                        interpolation=cv2.INTER_CUBIC)# * att_resized_back
                    
                    back = torch.from_numpy(cam_resized_back.copy()).unsqueeze(0)

                    cam_normalized_back = normalize_scoremap(cam_resized_back)

                    pred_mask = torch.cat([back, fore], dim=0)
                    pred_mask = t2n(torch.argmax(pred_mask, dim=0).squeeze())

                    ###save results####
                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)

                    cam_path = ospj(self.log_folder, 'masks', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), pred_mask)

                    cam_path = ospj(self.log_folder, 'classes', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), predict)

                    cam_path = ospj(self.log_folder, 'targets', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), target)

                    ###Evaluation
                    if 'mask' in self.eval_type:
                        eval_result = pred_mask #Ours_m
                    else:
                        eval_result = cam_normalized #Ours_p

                    if self.dataset_name == "OpenImages":
                        self.evaluator.accumulate(eval_result, image_id) #pIoU, PxAP
                    else:
                        self.evaluator.accumulate(eval_result, image_id, predict, target) # Top-1, GT-Known, BoxAcc
                    
                    if self.is_vis: #and count % 100 == 0: #visualization
                    
                        pred_mask = np.int64(pred_mask)
                        
                        vis_image = image.cpu().data * np.array(_IMAGENET_STDDEV).reshape([3, 1, 1]) + np.array(_IMAGENET_MEAN).reshape([3, 1, 1])
                        vis_image = np.int64(vis_image * 255)
                        vis_image[vis_image > 255] = 255
                        vis_image[vis_image < 0] = 0
                        vis_image = np.uint8(vis_image)

                        vis_path = ospj(self.log_folder, 'vis', image_id)
                        vis_seg_path = ospj(self.log_folder, 'vis_seg', image_id)
                        if not os.path.exists(ospd(vis_path)):
                            os.makedirs(ospd(vis_path))
                        if not os.path.exists(ospd(vis_seg_path)):
                            os.makedirs(ospd(vis_seg_path))
                        plt.imsave(ospj(vis_path)+"_fore.png", generate_vis(cam_normalized, vis_image).transpose(1, 2, 0))
                        plt.imsave(ospj(vis_path)+"_back.png", generate_vis(cam_normalized_back, vis_image).transpose(1, 2, 0))
                        plt.imsave(ospj(vis_seg_path)+"_oisseg.png", mark_boundaries(vis_image.transpose(1, 2, 0), pred_mask))
                    
            else:
                logits, cams = self.model(images, targets, return_cam=True)
                cams = t2n(cams)
                predicts = torch.argmax(logits, dim=1)
                predicts = t2n(predicts.view(predicts.shape[0], 1))
                targets = t2n(targets)


                for cam, image, image_id, predict, target in zip(cams, images, image_ids, predicts, targets):
                    cam_resized = cv2.resize(cam, image_size,
                                            interpolation=cv2.INTER_CUBIC)
                    cam_normalized = normalize_scoremap(cam_resized)

                    #predicts = torch.argmax(logits, dim=1)
                    #predicts = t2n(predicts.view(predicts.shape[0], 1))
                    #targets = t2n(targets)

                    if self.dataset_name == "OpenImages":
                        self.evaluator.accumulate(cam_normalized, image_id) #pIoU, PxAP
                    else:
                        self.evaluator.accumulate(cam_normalized, image_id, predict, target) # Top-1, GT-Known, BoxAcc

                                        
                    cam_path = ospj(self.log_folder, 'classes', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), predict)

                    cam_path = ospj(self.log_folder, 'targets', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), target)

                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                    
                    if self.is_vis:
                        ###
                        vis_image = image.cpu().data * np.array(_IMAGENET_STDDEV).reshape([3, 1, 1]) + np.array(_IMAGENET_MEAN).reshape([3, 1, 1])
                        vis_image = np.int64(vis_image * 255)
                        vis_image[vis_image > 255] = 255
                        vis_image[vis_image < 0] = 0
                        vis_image = np.uint8(vis_image)
                        vis_path = ospj(self.log_folder, 'vis', image_id)
                        if not os.path.exists(ospd(vis_path)):
                            os.makedirs(ospd(vis_path))
                        plt.imsave(ospj(vis_path)+".png", generate_vis(cam_normalized, vis_image).transpose(1, 2, 0))
                        ###

        if self.dataset_name == "OpenImages":
            return self.evaluator.compute()
        else:
            return self.evaluator.compute(), self.evaluator.compute_top1()



def generate_vis(p, img):
    # All the input should be numpy.array 
    # img should be 0-255 uint8

    C = 1
    H, W = p.shape

    prob = p

    prob[prob<=0] = 1e-7

    def ColorCAM(prob, img):
        C = 1
        H, W = prob.shape
        colorlist = []
        colorlist.append(color_pro(prob,img=img,mode='chw'))
        CAM = np.array(colorlist)/255.0
        return CAM

    #print(prob.shape, img.shape)
    CAM = ColorCAM(prob, img)
    #print(CAM.shape)
    return CAM[0, :, :, :]

def color_pro(pro, img=None, mode='hwc'):
	H, W = pro.shape
	pro_255 = (pro*255).astype(np.uint8)
	pro_255 = np.expand_dims(pro_255,axis=2)
	color = cv2.applyColorMap(pro_255,cv2.COLORMAP_JET)
	color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
	if img is not None:
		rate = 0.5
		if mode == 'hwc':
			assert img.shape[0] == H and img.shape[1] == W
			color = cv2.addWeighted(img,rate,color,1-rate,0)
		elif mode == 'chw':
			assert img.shape[1] == H and img.shape[2] == W
			img = np.transpose(img,(1,2,0))
			color = cv2.addWeighted(img,rate,color,1-rate,0)
			color = np.transpose(color,(2,0,1))
	else:
		if mode == 'chw':
			color = np.transpose(color,(2,0,1))	
	return color

def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam
