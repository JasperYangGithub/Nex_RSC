BACKBONE=resnet50

for TGT in {Augu,Sep,}
do
    python -u "/home/hcaoaf/github/RSC/Domain_Generalization/train.py" \
           --eval_mode \
           -c 9 \
           -e 50 \
           -b 128 \
           --cuda_number 0 \
           --target ${TGT}\
           --network ${BACKBONE} \
           --infer_model ./Domain_Generalization/save_models/tgt_Augu_src_batch_2-batch_3-Jul_epochs200_RSC_True_pretrained_False_fl_best.pth
done
