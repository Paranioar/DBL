CUDA_VISIBLE_DEVICES=0 python train.py \
--gpuid 0 --num_epochs 40 --lr_update 30 \
--save_path ./runs/baseline_f30k

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpuid 0 --data_name coco_precomp \
--num_epochs 20 --lr_update 10 --val_step 4000 \
--save_path ./runs/baseline_coco

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpuid 0 --num_epochs 30 --lr_update 0 \
--if_boost --boost_name boostabs \
--resume ./runs/baseline_f30k/checkpoint/model_best.pth.tar \
--save_path ./runs/f30k_abs

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpuid 0 --num_epochs 30 --lr_update 0 \
--if_boost --boost_name boostrel \
--resume ./runs/baseline_f30k/checkpoint/model_best.pth.tar \
--save_path ./runs/f30k_rel

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpuid 0 --data_name coco_precomp \
--num_epochs 20 --lr_update 10 --val_step 4000 \
--if_boost --boost_name boostabs \
--resume ./runs/baseline_coco/checkpoint/model_best.pth.tar \
--save_path ./runs/coco_abs

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpuid 0 --data_name coco_precomp \
--num_epochs 20 --lr_update 10 --val_step 4000 \
--if_boost --boost_name boostrel \
--resume ./runs/baseline_coco/checkpoint/model_best.pth.tar \
--save_path ./runs/coco_rel
