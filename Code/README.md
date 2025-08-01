# Let's Think Stage by Stage: Adapting Multi-stage Self-consistent Plan-and-Solve for ScienceQA

Think-ScienceQA leverages a lightweight semantic adapter to align textual and visual modalities effectively. In addition, a multi-stage self-consistent CoT mechanism offers structured guidance for generating optimal rationales. Furthermore, a two-stage fine-tuning strategy enables the LLMs to first generate structured reasoning, and then perform accurate answer inference.

![pipline](pipline.png)



## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Datetasets

Download the dataset from the following repository:

```
https://github.com/Think-ScienceQA/Think-ScienceQA.github.io/tree/master/data/
```

# Instructions

## Training 

Run the `run_training.sh` script.

### **Rationale Generation (Stage 1)**

```
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --model allenai/unifiedqa-t5-large \
  --user_msg rationale --img_type clip \
  --bs 1 --eval_bs 1 --output_len 512 \
  --epoch 20 --final_eval --prompt_format QCM-LE \
  --mode train --use_caption
```

### Fine-tune with Semantic Adapter (Stage 2)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --load_checkpoint /data/mm-cot-main/models/frozen_model_adapter/
    --model allenai/unifiedqa-t5-base \
    --user_msg rationale --img_type clip \
    --bs 1 --eval_bs 1 --eval_acc 10 --output_len 512 \
    --final_eval --epoch 50\
    --prompt_format QCM-LE\
    --mode train --use_caption 
```

### Answer Inference (Stage 3)

```
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg answer --img_type detr \
    --bs 8 --eval_bs 4 --eval_acc 10 --output_len 64 \
    --final_eval --prompt_format QCMG-A \
    --eval_le experiments/rationale_allenai-unifiedqa-t5-base_detr_QCM-E_lr5e-05_bs16_op512_ep20/predictions_ans_eval.json \
    --test_le experiments/rationale_allenai-unifiedqa-t5-base_detr_QCM-E_lr5e-05_bs16_op512_ep20/predictions_ans_test.json
```

## Inference 

Run the `run_inference.sh` script.

### **Generate Rationale**

```
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg rationale --img_type detr \
    --bs 8 --eval_bs 4 --eval_acc 10 --output_len 512 \
    --final_eval --prompt_format QCM-E \
    --evaluate_dir models/rationale

```

### answer inference

```
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg answer --img_type detr \
    --bs 8 --eval_bs 4 --eval_acc 10 --output_len 64 \
    --final_eval --prompt_format QCMG-A \
    --eval_le models/rationale/predictions_ans_eval.json \
    --test_le models/rationale/predictions_ans_test.json \
    --evaluate_dir models/answer
```