
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim

from config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer
from util import string_contains_any
import wsol
import wsol.method


def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
    }

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.','aggregator', 'classifier'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def __init__(self):
        self.cur_epoch = 0
        self.args = get_configs()
        seed = self.args.seed
        torch.manual_seed(seed)            # 为CPU设置随机种子
        torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        print(self.args.seed)
        #set_random_seed(self.args.seed)
        
        print(self.args)
        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter
        self.model = self._set_model()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = self._set_optimizer()
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class)

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['GT_Known_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]
        self._EVAL_METRICS += ['Top1_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        self._EVAL_METRICS += ['PxAP']
        self._EVAL_METRICS += ['pIoU']

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        model = wsol.__dict__[self.args.architecture](
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            pretrained_path=self.args.pretrained_path,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold,
            num_head = self.args.num_head)
        model = model.cuda()
        print(model)
        return model

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []
        param_features_bias = []
        param_classifiers_bias = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.model.named_parameters():

            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    if 'weight' in name:
                        print("1 times lr:", name)
                        param_features.append(parameter)
                    elif 'bias' in name:
                        print("2 times lr:", name)
                        param_features_bias.append(parameter)
                    
                elif self.args.architecture in ('resnet50'):
                    if 'weight' in name:
                        print("10 times lr:", name)
                        param_classifiers.append(parameter)
                    elif 'bias' in name:
                        print("20 times lr:", name)
                        param_classifiers.append(parameter)
            else:
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    if 'weight' in name:
                        print("10 times lr:", name)
                        param_classifiers.append(parameter)
                    elif 'bias' in name:
                        print("20 times lr:", name)
                        param_classifiers.append(parameter)
                elif self.args.architecture in ('resnet50'):
                    if 'weight' in name:
                        print("1 times lr:", name)
                        param_features.append(parameter)
                    elif 'bias' in name:
                        print("2 times lr:", name)
                        param_features_bias.append(parameter)
                    
        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.args.lr},
            {'params': param_features_bias, 'lr': self.args.lr * self.args.lr_bias_ratio},
            {'params': param_classifiers,
             'lr': self.args.lr * self.args.lr_classifier_ratio},
            {'params': param_classifiers_bias,
             'lr': self.args.lr * self.args.lr_bias_ratio * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        
        return optimizer


    def _wsol_training(self, images, target):
        
        if (self.args.wsol_method == 'cutmix' and
                self.args.cutmix_prob > np.random.rand(1) and
                self.args.cutmix_beta > 0):
            images, target_a, target_b, lam = wsol.method.cutmix(
                images, target, self.args.cutmix_beta)
            output_dict = self.model(images)
            logits = output_dict['logits']
            loss = (self.cross_entropy_loss(logits, target_a) * lam +
                    self.cross_entropy_loss(logits, target_b) * (1. - lam))
            return logits, loss
            
        if self.args.wsol_method == 'has':
            images = wsol.method.has(images, self.args.has_grid_size,
                                     self.args.has_drop_rate)

        output_dict = self.model(images, target)
        logits = output_dict['logits']

        if self.args.wsol_method in ('acol', 'spg'):
            loss = wsol.method.__dict__[self.args.wsol_method].get_loss(
                output_dict, target, spg_thresholds=self.args.spg_thresholds)
        elif self.args.wsol_method == 'bcam':
            loss = wsol.method.__dict__['bcam'].get_loss(
                output_dict, target, rate_ff=self.args.rate_ff, rate_fb=self.args.rate_fb, rate_bf=self.args.rate_bf, rate_bb=self.args.rate_bb)
        else:
            loss = self.cross_entropy_loss(logits, target)

        return logits, loss

    def train(self, split):
        self.model.train()
        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        for batch_idx, (images, target, _) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()

            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            logits, loss = self._wsol_training(images, target)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            num_correct += (pred == target).sum().item()
            num_images += images.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self):
        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}, current value: {}".format(
                        split, metric, current_performance))
                    if split != 'test':
                        print("Split {}, metric {}, best value: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_value))
                        print("Split {}, metric {}, best epoch: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_epoch))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, image_ids) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()
            output_dict = self.model(images)
            pred = output_dict['logits'].argmax(dim=1)

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        return classification_acc

    def evaluate(self, epoch, split):
        print("Evaluate epoch {}, split {}".format(epoch, split))
        self.model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy)

        cam_computer = CAMComputer(
            model=self.model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            log_folder=self.args.log_folder,
            wsol_method = self.args.wsol_method,
            target_layer = self.args.target_layer,
            is_vis = self.args.is_vis,
            eval_type = self.args.eval_type
        )
        cam_performance, cam_performance_top1 = cam_computer.compute_and_evaluate_cams()

        if self.args.multi_iou_eval or self.args.dataset_name == 'OpenImages':
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        self.performance_meters[split]['localization'].update(loc_score)

        if self.args.dataset_name in ('CUB', 'ILSVRC'):
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    'GT_Known_IOU_{}'.format(IOU_THRESHOLD)].update(
                    cam_performance[idx])
                self.performance_meters[split][
                    'Top1_IOU_{}'.format(IOU_THRESHOLD)].update(
                    cam_performance_top1[idx])
        else:
            self.performance_meters[split][
                    'PxAP'].update(
                    cam_performance_top1)
            self.performance_meters[split][
                    'pIoU'].update(
                    cam_performance)

    def _torch_save_model(self, filename, epoch):
        torch.save({'architecture': self.args.architecture,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch, split):
        self._torch_save_model(
            self._CHECKPOINT_NAME_TEMPLATE.format('%d'%(epoch)), epoch)
        print("saving checkpoint:", self._CHECKPOINT_NAME_TEMPLATE.format('%d'%(epoch)))
        if self.args.epochs == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)
            print("saving checkpoint:", self._CHECKPOINT_NAME_TEMPLATE.format('last'))

    def report_train(self, train_performance, epoch, split='train'):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        reporter_instance.add(
            key='{split}/classification'.format(split=split),
            val=train_performance['classification_acc'])
        reporter_instance.add(
            key='{split}/loss'.format(split=split),
            val=train_performance['loss'])
        reporter_instance.write()

    def report(self, epoch, split):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        for metric in self._EVAL_METRICS:
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].current_value)
            reporter_instance.add(
                key='{split}/{metric}_best'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].best_value)
        reporter_instance.write()

    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch % self.args.lr_decay_frequency == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = os.path.join(
            self.args.log_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))


def main():
    trainer = Trainer()

    
    print("===========================================================")
    
    if "test" in trainer.args.mode:
        checkpoint = torch.load(trainer.args.check_path)
        if trainer.args.dataset_name == "ILSVRC" or trainer.args.dataset_name == "OpenImages":
            from wsol.util import replace_layer
            checkpoint['state_dict'] = replace_layer(checkpoint['state_dict'], 'extractor_A', 'aggregator_A')
            checkpoint['state_dict'] = replace_layer(checkpoint['state_dict'], 'extractor_B', 'aggregator_B')
        trainer.model.load_state_dict(checkpoint['state_dict'], strict=True)

        if "debug" in trainer.args.mode:
            trainer.evaluate(trainer.args.epochs, split='test')
            trainer.print_performances()
            trainer.report(trainer.args.epochs, split='test')
            trainer.save_performances()
        else:
            trainer.evaluate(trainer.args.epochs, split='val')
            trainer.print_performances()
            trainer.report(trainer.args.epochs, split='val')
            trainer.save_performances()

            trainer.evaluate(trainer.args.epochs, split='test')
            trainer.print_performances()
            trainer.report(trainer.args.epochs, split='test')
            trainer.save_performances()

        return
    
    

    for epoch in range(trainer.args.epochs):
        print("===========================================================")
        print("Start epoch {} ...".format(epoch + 1))
        trainer.adjust_learning_rate(epoch + 1)
        trainer.cur_epoch = epoch
        train_performance = trainer.train(split='train')
        trainer.report_train(train_performance, epoch + 1, split='train')
        if (epoch + 1) % trainer.args.eval_frequency == 0:
            trainer.evaluate(epoch + 1, split='val')
            trainer.print_performances()
            trainer.report(epoch + 1, split='val')
            trainer.save_checkpoint(epoch + 1, split='val')
        print("Epoch {} done.".format(epoch + 1))

    print("===========================================================")
    print("Final epoch evaluation on test set ...")
    trainer.save_checkpoint(epoch + 1, split='val')
    trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
    trainer.evaluate(trainer.args.epochs, split='test')
    trainer.print_performances()
    trainer.report(trainer.args.epochs, split='test')
    trainer.save_performances()


if __name__ == '__main__':
    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
    main()
