import os

# delete env variables
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# NCCL_DEBUG=INFO
os.environ["NCCL_DEBUG"] = "INFO"

NUM_LABELS = 2
MODEL_NM = "microsoft/phi-2"
MAX_LEN = 768
LR = 2e-4
OUTPUT_DIR = 'phi-2-prompt-injection-QLoRA'
MAX_STEPS = 5000

# Define a function to change the label
def set_label_to_safe(example):
    example['label'] = 0
    return example

def set_label_to_unsafe(example):
    example['label'] = 1
    return example

# data loading
from datasets import load_dataset, concatenate_datasets

def load_and_append_data(safe, total_data, dataset_path, col_to_delete, prompt_col, train, val, test, filter):
    data = load_dataset(dataset_path)
    if filter != None:
        data = data.filter(filter)
    data['train'] = data[train]
    if val == None:
        _data = data['train'].train_test_split(train_size=0.9, seed=42)
        data['train'] = _data['train']
        data['val'] = _data['test']
    else:
        data['val'] = data[val]

    if test == None:
        _data = data['train'].train_test_split(train_size=0.8, seed=42)
        data['train'] = _data['train']
        data['test'] = _data['test']
    else:
        data['test'] = data[test]

    if safe == 'safe': 
        set_label = set_label_to_safe
    else:
        set_label = set_label_to_unsafe

    data['train'] = data['train'].map(set_label, remove_columns=col_to_delete)
    data['val'] = data['val'].map(set_label, remove_columns=col_to_delete)
    data['test'] = data['test'].map(set_label, remove_columns=col_to_delete)

    if prompt_col != 'prompt':
        data['train'] = data['train'].rename_column(prompt_col, 'prompt')
        data['val'] = data['val'].rename_column(prompt_col, 'prompt')
        data['test'] = data['test'].rename_column(prompt_col, 'prompt')

    # sample examples from the training data
    # samples if the dataset is large
    if len(data['train']) > 5000:
        data['train'] = data['train'].shuffle(seed=42).select(range(5000))
    if len(data['val']) > 100:
        data['val'] = data['val'].shuffle(seed=42).select(range(100))
    if len(data['test']) > 500:
        data['test'] = data['test'].shuffle(seed=42).select(range(500))

    if total_data == None:
        return data
    # combine the datasets
    total_data['train'] = concatenate_datasets([total_data.get('train'), data['train']])
    total_data['val'] = concatenate_datasets([total_data.get('val'), data['val']])
    total_data['test'] = concatenate_datasets([total_data.get('test'), data['test']])
    return total_data

# SAFE DATASETS
############################################################################################################
dataset_path = "HuggingFaceH4/no_robots"
col_to_delete = ['prompt_id', 'messages', 'category']
prompt_col = 'prompt'
total_data = load_and_append_data('safe', None, dataset_path, col_to_delete, prompt_col, 'train_sft', None, 'test_sft', None)
############################################################################################################
dataset_path = "Dahoas/synthetic-hh-rlhf-prompts"
col_to_delete = []
prompt_col = 'prompt'
total_data = load_and_append_data('safe', total_data, dataset_path, col_to_delete, prompt_col, 'train', None, None, None)
############################################################################################################
dataset_path = "HuggingFaceH4/ultrachat_200k"
col_to_delete = ['prompt_id', 'messages']
prompt_col = 'prompt'
total_data = load_and_append_data('safe', total_data, dataset_path, col_to_delete, prompt_col, 'train_sft', None, 'test_sft', None)
############################################################################################################
dataset_path = "HuggingFaceH4/instruction-dataset"
col_to_delete = ['completion', 'meta']
prompt_col = 'prompt'
total_data = load_and_append_data('safe', total_data, dataset_path, col_to_delete, prompt_col, 'test', None, None, None)
############################################################################################################

# UNSAFE DATASETS
############################################################################################################
dataset_path = "Lakera/gandalf_ignore_instructions"
col_to_delete = ['similarity']
prompt_col = 'text'
total_data = load_and_append_data('unsafe', total_data, dataset_path, col_to_delete, prompt_col, 'train', 'validation', 'test', None)
############################################################################################################
dataset_path = "imoxto/prompt_injection_cleaned_dataset-v2"
col_to_delete = ['model', 'labels']
prompt_col = 'text'
filter = lambda example: example['labels'] == 1
total_data = load_and_append_data('unsafe', total_data, dataset_path, col_to_delete, prompt_col, 'train', None, None, None)
############################################################################################################
dataset_path = "hackaprompt/hackaprompt-dataset"
col_to_delete = ['level', 'prompt', 'completion', 'model', 'expected_completion', 'token_count', 'correct', 'error', 'score', 'dataset', 'timestamp', 'session_id']
prompt_col = 'user_input'
filter = lambda example: example['correct'] == True
total_data = load_and_append_data('unsafe', total_data, dataset_path, col_to_delete, prompt_col, 'train', None, None, filter)
############################################################################################################
dataset_path = "rubend18/ChatGPT-Jailbreak-Prompts"
col_to_delete = ['Name', 'Votes', 'Jailbreak Score', 'GPT-4']
prompt_col = 'Prompt'
total_data = load_and_append_data('unsafe', total_data, dataset_path, col_to_delete, prompt_col, 'train', None, None, None)
############################################################################################################

# setting weights for loss function to address class imbalance
data = total_data
pos_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().label.value_counts()[1])
neg_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().label.value_counts()[0])

# print the sample size
print(f'Total samples: {len(data["train"])}')
print(f'Safe samples: {data["train"].to_pandas().label.value_counts()[0]}')
print(f'Unsafe samples: {data["train"].to_pandas().label.value_counts()[1]}')

# print the weights
print(f'Positive weights: {pos_weights}')
print(f'Negative weights: {neg_weights}')

# tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NM, trust_remote_code=True)
# set pad token - to avoid error while training
tokenizer.pad_token = tokenizer.eos_token

def preprocessing_function(examples):
    return tokenizer(examples['prompt'], truncation=True, max_length=MAX_LEN)

# Apply the preprocessing function and remove the undesired columns
tokenized_datasets = data.map(preprocessing_function, batched=True)
# Set to torch format
tokenized_datasets.set_format("torch")

#  It takes a batch of examples and ensures that each sequence in the batch has the same length
#  by padding the shorter ones. Ensures fixed-size input sequences
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from torch import nn

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_use_double_quant=False,
    )

basemodel = AutoModelForCausalLM.from_pretrained(MODEL_NM, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True, torch_dtype=torch.bfloat16)
basemodel.config.pretraining_tp = 1
basemodel.config.pad_token_id = tokenizer.pad_token_id

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

class PhiPreTrainedModel(PreTrainedModel):
    config_class = basemodel.config_class
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


#custom class - modified from PhiForSequenceClassification
class PhiForSequenceClassificationModified(PhiPreTrainedModel):
    def __init__(self, basemodel):
        super().__init__(basemodel.config)
        self.num_labels = NUM_LABELS#changed
        self.model = basemodel.model#changed
        self.score = nn.Linear(basemodel.config.hidden_size, NUM_LABELS, bias=False)#changed

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward("PHI_INPUTS_DOCSTRING")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids,
            attention_mask = attention_mask,

        )

        hidden_states = model_outputs[0]#changed
        logits = self.score(hidden_states)


        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        loss = None
        if not return_dict:
            output = (pooled_logits,) + model_outputs[1:]
            print(output)
            if loss is not None:
              print('Loss is not none')
              return ((loss,) + output)
            else:
              return output
            
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )#changed

# ⚠️ Temporary Patch
# We call peft for CAUSAL_LM first and then for SEQ_CLS. This is needed to train the last scoring layer for PhiForSequenceClassificationModified.
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
############################################################################################################

import evaluate
import numpy as np
from transformers import Trainer,TrainingArguments
def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weights, pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=200,
    save_total_limit =2,
    logging_steps=10,
    learning_rate=LR,
    bf16=True,
    max_grad_norm=.3,
    max_steps=MAX_STEPS,
    warmup_ratio=.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to='none',
    evaluation_strategy="steps",# Evaluate the model every specified number of steps
    eval_steps=200,
)

# uncomment the following lines to use wandb
# import wandb
# wandb.init(project=OUTPUT_DIR)

phi2_trainer = WeightedCELossTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

phi2_trainer.train()