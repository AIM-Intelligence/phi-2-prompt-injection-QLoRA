# phi-2-prompt-injection-QLoRA
- Training code for Prompt Injection Detection Model with LoRA Fine-tuning phi-2

- phi-2 used for sequence classification task

- reference: https://medium.com/@noufalsamsudin/microsoft-phi-2-for-classification-b83beaec2069

- Fix to fit current implementation of phi-2 in huggingface. (03/07/2024)

- https://huggingface.co/ysy970923/phi-2-prompt-injection-QLoRA

## How to use
- Fix the parameters as needed in each files
- Run train.py to train the model
    ```python
    python train.py
    ```
- Run test_on_custom_dataset.py to test the model on custom dataset
- modify the custom dataset to add your own data for testing
    ```python
    python test_on_custom_dataset.py
    ```

## Files
- train.py: Training code
- test_on_custom_dataset.py: Test on custom dataset

## Resource Requirements
- 1x NVIDIA 4090 GPU

## Note
- ⚠️ Temporary Patch: In train.py, we call get_peft_model twice.
```python
# ⚠️ Temporary Patch
# We call peft for CAUSAL_LM first and then for SEQ_CLS. This is needed to to train the last scoring layer for PhiForSequenceClassificationModified.
# If possible, Please suggest a better way to do this.
############################################################################################################
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

peft_model = get_peft_model(basemodel, LoraConfig(
    task_type= TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules = ['q_proj', 'k_proj', 'v_proj'],
    bias='none'
))

peft_model.print_trainable_parameters()

model = PhiForSequenceClassificationModified(basemodel)

peft_model = get_peft_model(model, LoraConfig(
    task_type= TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules = ['q_proj', 'k_proj', 'v_proj'],
    bias='none'
))

peft_model.print_trainable_parameters()
peft_model = peft_model.to(device='cuda')
```

