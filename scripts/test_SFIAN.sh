GPU_IDS=0

DATAROOT='/home/hui/Research/dataset/crack500'

NAME=SFIAN
MODEL=SFIAN
DATASET_MODE=SFIAN

LOAD_WIDTH=256
LOAD_HEIGHT=256
BATCH_SIZE=1
NORM=batch

NUM_CLASSES=1
NUM_TEST=10

python3 test.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --batch_size ${BATCH_SIZE} \
  --num_classes ${NUM_CLASSES} \
  --load_width ${LOAD_WIDTH} \
  --load_height ${LOAD_HEIGHT} \
  --norm ${NORM} \
  --num_test ${NUM_TEST}\
  --display_sides 1

