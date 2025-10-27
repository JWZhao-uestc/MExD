# Ctrans
# TCGA
python train_moe.py --num_classes 2 --dataset TCGA --agg no --model moe --feat_type ctrans --trained_model your_trained_model_without_moe
# C16
python train_moe.py --num_classes 2 --dataset Camelyon16 --agg no --model moe --feat_type ctrans --trained_model your_trained_model_without_moe
# BRACS
python train_moe.py --num_classes 3 --dataset BRACS --agg no --model moe --feat_type ctrans --trained_model your_trained_model_without_moe

# ViT
# TCGA
python train_moe.py --num_classes 2 --dataset TCGA --agg no --model moe --feat_type vit --trained_model your_trained_model_without_moe
# C16
python train_moe.py --num_classes 2 --dataset Camelyon16 --agg no --model moe --feat_type vit --trained_model your_trained_model_without_moe
# BRACS
python train_moe.py --num_classes 3 --dataset BRACS --agg no --model moe --feat_type vit --trained_model your_trained_model_without_moe
