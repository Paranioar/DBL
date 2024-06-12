CUDA_VISIBLE_DEVICES=0 python train.py \
--save_path ./runs/f30k_abs \
--num_epochs 40 --lr_update 30 --num_branch 2 \
--gpuid 0 --boost_name boostabs

CUDA_VISIBLE_DEVICES=0 python train.py \
--save_path ./runs/f30k_rel \
--num_epochs 40 --lr_update 30 --num_branch 2 \
--gpuid 0 --boost_name boostrel

CUDA_VISIBLE_DEVICES=0 python train.py \
--save_path ./runs/coco_abs --data_name coco_precomp \
--num_epochs 20 --lr_update 10 --num_branch 2 \
--gpuid 0 --boost_name boostabs

CUDA_VISIBLE_DEVICES=0 python train.py \
--save_path ./runs/coco_rel --data_name coco_precomp \
--num_epochs 20 --lr_update 10 --num_branch 2 \
--gpuid 0 --boost_name boostrel


