import argparse
import json
import os

import numpy as np
import torch
import wandb
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from train_utils import fix_model, fix_tokenizer, set_random_seed, SUPPORTED_DATASETS
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          DataCollatorForTokenClassification, EvalPrediction,
                          T5ForConditionalGeneration, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils.instruct_dataset import InstructDataset, Instruction
from metric import calculate_metrics, extract_classes


# https://github.com/huggingface/peft/issues/96
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control

def train(
    train_instructions: list[Instruction],
    test_instructions: list[Instruction],
    model_type: str,
    dataset_name: str,
    output_dir: str,
    seed: int,
    config_file: str,
    push_to_hub: bool
):
    set_random_seed(seed)
    with open(config_file, "r") as r:
        config = json.load(r)

    lora_config = config.get("lora")
    model_name = config['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer)

    # def preprocess_logits_for_metrics(logits, labels):
    #     """
    #     Original Trainer may have a memory leak.
    #     This is a workaround to avoid storing too many tensors that are not needed.
    #     """
    #     pred_ids = torch.argmax(logits[0], dim=-1)
    #     return pred_ids, labels

    # def compute_metrics(eval_prediction: EvalPrediction, tokenizer=tokenizer):
    #     #predictions = np.argmax(eval_prediction.predictions, axis=-1)
    #     #labels = eval_prediction.label_ids

    #     predictions = eval_prediction.predictions
    #     labels = eval_prediction.labels

    #     print("predictions", predictions.shape, flush=True)
    #     print("labels", labels.shape, flush=True)

    #     extracted_entities = []
    #     target_entities = []
    #     for ind, pred in enumerate(predictions):
    #         non_masked_indices = (labels[ind] != -100)
    #         pred = tokenizer.decode(pred, skip_special_tokens=True)
    #         label = tokenizer.decode(labels[ind][non_masked_indices], skip_special_tokens=True)

    #         extracted_entities.append(extract_classes(pred))
    #         target_entities.append(extract_classes(label))

    #     return calculate_metrics(extracted_entities, target_entities, return_only_f1=True)

    only_target_loss = config.get("only_target_loss", True)
    max_source_tokens_count = config["max_source_tokens_count"]
    max_target_tokens_count = config["max_target_tokens_count"]

    train_dataset = InstructDataset(
        train_instructions,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        model_type=model_type,
        only_target_loss=only_target_loss
    )

    val_dataset = InstructDataset(
        test_instructions,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        model_type=model_type,
        only_target_loss=only_target_loss
    )

    model_classes = {
        'llama': {
            'data_collator': DataCollatorForTokenClassification,
            'model': AutoModelForCausalLM
        },
        'bloomz': {
            'data_collator': DataCollatorForTokenClassification,
            'model': AutoModelForCausalLM
        },
        'marx': {
            'data_collator': DataCollatorForTokenClassification,
            'model': AutoModelForCausalLM
        },
        'mistral': {
            'data_collator': DataCollatorForTokenClassification,
            'model': AutoModelForCausalLM
        },
        't5': {
            'data_collator': DataCollatorForSeq2Seq,
            'model': T5ForConditionalGeneration
        }
    }
    data_collator = model_classes[model_type]['data_collator'](tokenizer, pad_to_multiple_of=8)

    load_in_8bit = bool(config.get("load_in_8bit", True))
    is_adapter = config['is_adapter']
    if load_in_8bit:
        if is_adapter:
            peft_config = PeftConfig.from_pretrained(model_name)
            model = model_classes[model_type]['model'].from_pretrained(
                peft_config.base_model_name_or_path,
                load_in_8bit=True,
                device_map='auto',
                #use_flash_attention_2=True
            )
            model = fix_model(model, tokenizer, use_resize=False)
            model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(model, model_name, is_trainable=True)
        else:
            model = model_classes[model_type]['model'].from_pretrained(
                model_name,
                #load_in_8bit=True,
                device_map='cuda:0',
                #use_flash_attention_2=True
            )
            model = fix_model(model, tokenizer, use_resize=False)
            #model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = fix_model(model, tokenizer, use_resize=False)

    # Default model generation params
    model.config.num_beams = 5
    max_tokens_count = max_target_tokens_count + max_source_tokens_count + 1
    model.config.max_length = max_tokens_count

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    deepspeed_config = config.get("deepspeed")
    trainer_config = config["trainer"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        #save_total_limit=8,
        #load_best_model_at_end=True,
        report_to='wandb',
        ddp_find_unused_parameters=None,
        deepspeed=deepspeed_config,
        **trainer_config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[SavePeftModelCallback],
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
        #preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )

    with wandb.init(project="Instruction NER") as run:
        #model.print_trainable_parameters()
        trainer.train()
        torch.save(model.state_dict(), f"final_{model_type}.pth")
        if 'llama2' in config_file:
            model_type = 'llama2'
        if push_to_hub:
            model.push_to_hub(f"poteminr/{model_type}-{dataset_name}", use_auth_token=True)
            tokenizer.push_to_hub(f"poteminr/{model_type}-{dataset_name}", use_auth_token=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='conll2003', type=str, help='name of dataset')
    parser.add_argument("--data_path", default='data/rudrec/rudrec_annotated.json', type=str, help='train file_path')
    parser.add_argument("--output_dir", default='marx_model_32/', type=str, help='output_dir')
    parser.add_argument("--test_size", default=0.3, type=float, help='test_size')
    parser.add_argument("--random_seed", default=1719, type=int, help='random_seed')
    parser.add_argument("--config_file", default='configs/marx-3b_lora.json', type=str, help='path to config file')
    parser.add_argument("--model_type", default='marx', type=str, help='model type')
    parser.add_argument("--max_instances", default=-1, type=int, help='max number of instructions')
    parser.add_argument("--push_to_hub", default=False, type=bool, help='push to hugginface hub')
    arguments = parser.parse_args()

    assert arguments.dataset_name in SUPPORTED_DATASETS, f'expected dataset name from {SUPPORTED_DATASETS}'

    if arguments.dataset_name == 'rudrec':
        from utils.rudrec.rudrec_reader import create_train_test_instruct_datasets
        train_dataset, test_dataset = create_train_test_instruct_datasets(
            data_path=arguments.data_path,
            max_instances=arguments.max_instances,
            test_size=arguments.test_size,
            random_seed=arguments.random_seed
        )
    elif arguments.dataset_name =='nerel_bio':
        from utils.nerel_bio.nerel_reader import create_instruct_dataset
        train_path = os.path.join(arguments.data_path, 'train')
        test_path = os.path.join(arguments.data_path, 'test')
        train_dataset = create_instruct_dataset(train_path, max_instances=arguments.max_instances)
        test_dataset = create_instruct_dataset(test_path, max_instances=arguments.max_instances)
    else:
        from utils.conll2003.conll_reader import create_instruct_dataset
        train_dataset = create_instruct_dataset(split='train', max_instances=arguments.max_instances)
        test_dataset = create_instruct_dataset(split='validation', max_instances=arguments.max_instances)

    train(
        train_instructions=train_dataset,
        test_instructions=test_dataset,
        model_type=arguments.model_type,
        dataset_name=arguments.dataset_name,
        output_dir=arguments.output_dir,
        seed=arguments.random_seed,
        config_file=arguments.config_file,
        push_to_hub=arguments.push_to_hub
    )
