export EXP_DIR=./results
export N_STEPS=500 #1000
export RUN_NAME=run_2
export PRIOR_TYPE=f_phi_prior
export CAT_F_PHI=_cat_f_phi
export F_PHI_TYPE=f_phi_supervised  #f_phi_self_supervised
export MODEL_VERSION_DIR='your directory name'
export LOSS=card_onehot_conditional
export TASK=TCGA-WSI-vit ######
export N_SPLITS=1
export DEVICE_ID=1
export N_THREADS=8
export FEATURE_EXACTOR=transmil

# train
python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --model ${FEATURE_EXACTOR} --config configs/${TASK}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
