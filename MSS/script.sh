CUDA_VISIBLE_DEVICES=0 python train.py \
--save_path ./runs/f30k_abs \
--num_epochs 40 --lr_update 30 --momentum_teacher 0.99995 \
--gpuid 0 --boost_name boostabs

CUDA_VISIBLE_DEVICES=0 python train.py \
--save_path ./runs/f30k_rel \
--num_epochs 40 --lr_update 30 --momentum_teacher 0.99995 \
--gpuid 0 --boost_name boostrel

CUDA_VISIBLE_DEVICES=0 python train.py \
--save_path ./runs/coco_abs \
--num_epochs 20 --lr_update 10 --momentum_teacher 0.99995 \
--gpuid 0 --boost_name boostabs \
--data_name coco_precomp --val_step 4000

CUDA_VISIBLE_DEVICES=0 python train.py \
--save_path ./runs/coco_rel \
--num_epochs 20 --lr_update 10 --momentum_teacher 0.99995 \
--gpuid 0 --boost_name boostrel \
--data_name coco_precomp --val_step 4000