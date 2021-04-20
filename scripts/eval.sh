BACKBONE=resnet50

for TGT in {Nov,}
do
    python -u "/home/hcaoaf/github/RSC/Domain_Generalization/train.py" \
           --eval_mode \
           -c 9 \
           -e 50 \
           -b 128 \
           --cuda_number 0 \
           --target ${TGT}\
           --imbalance_ratio 100\
           --network ${BACKBONE} \
           --infer_model ./Domain_Generalization/save_models/tgt_Augu_src_batch_2-Jul_epochs200_RSC_True_pretrained_False_fl_best.pth
done
