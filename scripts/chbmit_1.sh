# export CUDA_VISIBLE_DEVICES=1

# python main_1.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 8  --mode train --dataset CHBMIT  --data_path /data/lixinying/chb-mit_18ch/ --input_c 18  --output_c 18 --win_size 256 --step 32
python main_3.py --anormly_ratio 0.5 --num_epochs 3 --batch_size 16  --mode test  --dataset CHBMIT   --data_path /data/lixinying/chb-mit_18ch/  --input_c 18    --output_c 18 --win_size 256 --step 32 --pretrained_model 20


# # export CUDA_VISIBLE_DEVICES=1

# python main.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 8  --mode train --dataset CHBMIT  --data_path /data/lixinying/chb-mit_18ch/ --input_c 18  --output_c 18 --win_size 256 --step 32
# python main.py --anormly_ratio 0.5 --num_epochs 3 --batch_size 16  --mode test  --dataset CHBMIT   --data_path /data/lixinying/chb-mit_18ch/  --input_c 18    --output_c 18 --win_size 256 --step 32 --pretrained_model 20


