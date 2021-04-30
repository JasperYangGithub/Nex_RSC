ARCH=resnet50
SRC=Nex_trainingset
TGT=Jan2021
CLS_NUM=9
EPOCH=200
BATCH_SIZE=32

LOSS=fl
PRETRAINED=True
RSC=True

GPU=1

mkdir -p Domain_Generalization/explog
LOG_PATH=Domain_Generalization/explog

python -u /home/hcaoaf/github/RSC/Domain_Generalization/train.py \
       -c ${CLS_NUM} \
       -e ${EPOCH} \
       -b ${BATCH_SIZE} \
       --downsample_target \
       --cuda_number ${GPU} \
       --save_model \
       --source ${SRC} \
       --target ${TGT}\
       --network ${ARCH} \
       --RSC_flag \
       --loss fl \
       >>${LOG_PATH}/${TGT}_BASED_ON_${SRC}_RSC_${RSC}_epochs${EPOCH}_${ARCH}_Pretrained_${PRETRAINED}_loss_${LOSS}.log

