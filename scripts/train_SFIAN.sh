GPU_IDS=0

DATAROOT='/home/hui/Research/dataset/crack500'

NAME=SFIAN
MODEL=SFIAN
DATASET_MODE=SFIAN
LOAD_WIDTH=256
LOAD_HEIGHT=256

BATCH_SIZE=1
NUM_CLASSES=1
LOSS_MODE=focal

NORM=batch
NITER=60
NITER_DECAY=20


python3 train.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --batch_size ${BATCH_SIZE} \
  --load_width ${LOAD_WIDTH} \
  --load_height ${LOAD_HEIGHT} \
  --num_classes ${NUM_CLASSES} \
  --norm ${NORM} \
  --lr_decay_iters 40 \
  --lr_policy step \
  --no_flip 0 \
  --display_id 0 \
  --loss_mode ${LOSS_MODE}