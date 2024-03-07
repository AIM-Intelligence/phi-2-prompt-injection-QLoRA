import os

# delete env variables
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# NCCL_DEBUG=INFO
os.environ["NCCL_DEBUG"] = "INFO"

NUM_LABELS = 2
MODEL_NM = "microsoft/phi-2"
MAX_LEN = 768
LR = 2e-4
OUTPUT_DIR = 'phi-2-prompt-injection-QLoRA'
# test various checkpoints
CHECKPOINT = 'checkpoint-4800'
PUSH_TO_HUB = False

# tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NM, trust_remote_code=True,)
tokenizer.pad_token = tokenizer.eos_token

def preprocessing_function(examples):
    return tokenizer(examples['prompt'], truncation=True, max_length=MAX_LEN)

#  It takes a batch of examples and ensures that each sequence in the batch has the same length
#  by padding the shorter ones. Ensures fixed-size input sequences
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from torch import nn
from datasets import Dataset, concatenate_datasets

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_use_double_quant=False,
    )

# load model in half precision - not working for training
# basemodel = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", device_map="cuda", trust_remote_code=True)
# loads model in full precision - works
basemodel = AutoModelForCausalLM.from_pretrained(MODEL_NM, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True, torch_dtype=torch.bfloat16)
basemodel.config.pretraining_tp = 1
basemodel.config.pad_token_id = tokenizer.pad_token_id
basemodel.model

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
    
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
# model = prepare_model_for_kbit_training(model)

peft_model = get_peft_model(basemodel, LoraConfig(
    task_type= TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules = ['q_proj', 'k_proj', 'v_proj'],
    bias='none'
))

peft_model.print_trainable_parameters()
peft_model = peft_model.to(device='cuda')

prompt = """I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is "I am in Istanbul/Beyoğlu and I want to visit only museums.\""""
inputs = tokenizer(prompt, truncation=True, max_length=MAX_LEN, return_tensors="pt")
inputs = inputs.to('cuda')
# chagne input to float16
outputs = peft_model(**inputs)


model = PhiForSequenceClassificationModified(basemodel)

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

peft_model = get_peft_model(model, LoraConfig(
    task_type= TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.5,
    target_modules = ['q_proj', 'k_proj', 'v_proj'],
    bias='none'
))

peft_model.print_trainable_parameters()
peft_model = peft_model.to(device='cuda')

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

class CELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=100,
    save_total_limit =2,
    logging_steps=10,
    learning_rate=LR,
    # fp16=True,
    bf16=True,
    max_grad_norm=.3,
    warmup_ratio=.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to='none',
    evaluation_strategy="steps",# Evaluate the model every specified number of steps
    eval_steps=100,
)

# read "jailbreak.json"
import json

with open('./custom_dataset.json') as f:
    data = json.load(f)

jailbreak_triggers = data['jailbreak']

for i in range(len(jailbreak_triggers)):
    if "[INSERT PROMPT HERE]" not in jailbreak_triggers[i]:
        # add [INSERT PROMPT HERE] to the end of the trigger
        jailbreak_triggers[i] = jailbreak_triggers[i] + " [INSERT PROMPT HERE]"

injected_goals = ["What is the password?"]

# combine jailbreak prompts and harmful goals
jailbreak_prompts = []
for jailbreak_trigger in jailbreak_triggers:
    for harmful_goal in injected_goals:
        # replace [INSERT PROMPT HERE] with harmful_goal
        jailbreak_prompts.append(jailbreak_trigger.replace("[INSERT PROMPT HERE]", harmful_goal))

test_set = jailbreak_prompts
labels = [1 for _ in range(len(test_set))]
# list to tokenized dataset
test_set = Dataset.from_dict({'prompt': test_set, 'label': labels})
# safe set
safe_data = data['non-jailbreak']
labels = [0 for _ in range(len(safe_data))]
# list to tokenized dataset
safe_set = Dataset.from_dict({'prompt': safe_data, 'label': labels})
# combine safe and harmful data
test_set = concatenate_datasets([test_set, safe_set])
# test_set = safe_set
# preprocess
tokenized_test_set = test_set.map(preprocessing_function, batched=True)

phi2_trainer = CELossTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=None,
    eval_dataset=tokenized_test_set,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

import os
best_model_ckpt = os.path.join(OUTPUT_DIR, CHECKPOINT)

from peft import PeftModel
loaded_model = PeftModel.from_pretrained(model, best_model_ckpt, is_trainable=False)

phi2_trainer.model = loaded_model

# print score layer weight
print(loaded_model.model.score.weight)

# evaluate
print(phi2_trainer.evaluate())

if PUSH_TO_HUB:
    # push to huggingface
    phi2_trainer.push_to_hub(OUTPUT_DIR)
