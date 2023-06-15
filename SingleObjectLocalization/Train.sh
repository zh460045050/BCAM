#eval_type="mask"
eval_type="prob"
#####ResNet
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name CUB \
               --architecture resnet50 \
               --wsol_method "bcam" \
               --experiment_name resnet \
               --pretrained TRUE \
               --num_val_sample_per_class 0 \
               --large_feature_map TRUE \
               --batch_size 32 \
               --epochs 20 \
               --lr 1.70E-4 \
               --lr_decay_frequency 15 \
               --weight_decay 1.00E-04 \
               --override_cache FALSE \
               --workers 16 \
               --box_v2_metric True \
               --iou_threshold_list 30 50 70 \
               --eval_checkpoint_type last \
               --rate_ff 1 \
               --rate_fb 1 \
               --rate_bf 1 \
               --rate_bb 1 \
               --save_dir 'train_log_prob' \
               --seed 4 \
               --target_layer "layer4" \
               --eval_type $eval_type \
               --eval_frequency 20

#######VGG
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name CUB \
               --architecture vgg16 \
               --wsol_method "bcam" \
               --experiment_name vgg \
               --pretrained TRUE \
               --num_val_sample_per_class 0 \
               --large_feature_map FALSE \
               --batch_size 32 \
               --epochs 20 \
               --lr 1.7e-5 \
               --lr_decay_frequency 15 \
               --weight_decay 5.00E-04 \
               --override_cache FALSE \
               --workers 16 \
               --box_v2_metric True \
               --iou_threshold_list 30 50 70 \
               --eval_checkpoint_type last \
               --rate_ff 1 \
               --save_dir 'train_log_prob' \
               --seed 10 \
               --rate_fb 0.4 \
               --rate_bf 0.4 \
               --rate_bb 0.2 \
               --num_head 100 \
               --target_layer "conv6" \
               --eval_type $eval_type \
               --eval_frequency 20

######Inception
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name CUB \
               --architecture inception_v3 \
               --wsol_method "bcam" \
               --experiment_name inception \
               --pretrained TRUE \
               --num_val_sample_per_class 0 \
               --large_feature_map TRUE \
               --batch_size 32 \
               --epochs 50 \
               --lr 1.7E-3 \
               --lr_decay_frequency 15 \
               --lr_bias_ratio 2 \
               --weight_decay 5.00E-04 \
               --override_cache FALSE \
               --workers 16 \
               --box_v2_metric True \
               --iou_threshold_list 30 50 70 \
               --eval_checkpoint_type last \
               --rate_ff 1 \
               --rate_fb 0.4 \
               --rate_bf 0.4 \
               --rate_bb 0.2 \
               --num_head 100 \
               --save_dir 'train_log_prob' \
               --seed 25 \
               --target_layer "SPG_A3_1b" \
               --eval_frequency 50 \
               --eval_type $eval_type


