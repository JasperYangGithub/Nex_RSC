ARCH=resnet50
SRC=Sep-Oct-Nov
TGT=Dec
CLS_NUM=9
EPOCH=200
BATCH_SIZE=32

PRETRAINED=False
RSC=False
GPU=8

mkdir -p Domain_Generalization/explog
LOG_PATH=Domain_Generalization/explog

python -u /home/hcaoaf/github/RSC/Domain_Generalization/train.py \
       -c ${CLS_NUM} \
       -e ${EPOCH} \
       -b ${BATCH_SIZE} \
       --cuda_number ${GPU} \
       --save_model \
       --source ${SRC} \
       --target ${TGT}\
       --network ${ARCH} \
       >>${LOG_PATH}/${TGT}_BASED_ON_${SRC}_RSC_${RSC}_epoch${EPOCH}_${ARCH}_Pretrained_${PRETRAINED}.log
