# rationale generation
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg rationale --img_type detr \
    --bs 8 --eval_bs 4 --eval_acc 10 --output_len 512 \
    --final_eval --prompt_format QCM-E
    
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --load_checkpoint /data/mm-cot-main/models/frozen_model_adapter/
    --model allenai/unifiedqa-t5-base \
    --user_msg rationale --img_type clip \
    --bs 1 --eval_bs 1 --eval_acc 10 --output_len 512 \
    --final_eval --epoch 50\
    --prompt_format QCM-LE\
    --mode train --use_caption 

# answer inference
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg answer --img_type detr \
    --bs 8 --eval_bs 4 --eval_acc 10 --output_len 64 \
    --final_eval --prompt_format QCMG-A \
    --eval_le experiments/rationale_allenai-unifiedqa-t5-base_detr_QCM-E_lr5e-05_bs16_op512_ep20/predictions_ans_eval.json \
    --test_le experiments/rationale_allenai-unifiedqa-t5-base_detr_QCM-E_lr5e-05_bs16_op512_ep20/predictions_ans_test.json