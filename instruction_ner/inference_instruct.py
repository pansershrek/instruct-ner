import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, GenerationConfig
from peft import PeftConfig, PeftModel

from metric import extract_classes
from train_utils import SUPPORTED_DATASETS

from peft import (
    LoraConfig, PeftConfig, PeftModel, get_peft_model,
    prepare_model_for_kbit_training
)

def batch(iterable, n=4):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='conll2003', type=str, help='name of dataset')
    parser.add_argument("--data_path", default='data/rudrec/rudrec_annotated.json', type=str, help='train file_path')
    parser.add_argument("--model_type", default='marx', type=str, help='model type')
    parser.add_argument("--model_name", default='acrastt/Marx-3B-V2', type=str, help='model name from hf')
    parser.add_argument("--prediction_path", default='marx_prediction.json', type=str, help='path for saving prediction')
    parser.add_argument("--max_instances", default=-1, type=int, help='max number of instruction')
    parser.add_argument("--batch_size", default=4, type=int, help='number of instructions in batch')
    parser.add_argument("--peft_config", default="/home/admin/instruct-ner/instruction_ner/marx_models/checkpoint-872", type=str, help="Path to peft config")
    arguments = parser.parse_args()

    # assert arguments.dataset_name in SUPPORTED_DATASETS, f'expected dataset name from {SUPPORTED_DATASETS}'

    # model_name = arguments.model_name
    # generation_config = GenerationConfig.from_pretrained(model_name)

    # peft_config = PeftConfig.from_pretrained(arguments.model_name)
    # base_model_name = peft_config.base_model_name_or_path

    # models = {'llama': AutoModelForCausalLM, 't5': T5ForConditionalGeneration}

    # model = models[arguments.model_type].from_pretrained(
    #     base_model_name,
    #     load_in_8bit=True,
    #     device_map='auto'
    # )

    # model = PeftModel.from_pretrained(model, model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        arguments.model_name,
        load_in_8bit=True,
        device_map='cuda:0',
        #n_gpu_layers = 35,
        #n_ctx=2048,
        #n_parts=1,
        #use_mmap=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(arguments.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(
        model,
        arguments.peft_config,
    )
    #model = model.merge_and_unload()

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        early_stopping = True,
        no_repeat_ngram_size = 3,
        penalty_alpha = 0.6,
        #epsilon_cutoff = 0.1,
        length_penalty = -10,
        eos_token_id=tokenizer.eos_token_id
    )

    model.eval()
    model = torch.compile(model)

    #if torch.cuda.device_count() > 1:
    #    model.is_parallelizable = True
    #    model.model_parallel = True

    if arguments.dataset_name == 'rudrec':
        from utils.rudrec.rudrec_reader import create_train_test_instruct_datasets
        from utils.rudrec.rudrec_utis import ENTITY_TYPES

        _, test_dataset = create_train_test_instruct_datasets(arguments.data_path)
        if arguments.max_instances != -1 and arguments.max_instances < len(test_dataset):
            test_dataset = test_dataset[:arguments.max_instances]
    elif arguments.dataset_name == 'nerel_bio':
        from utils.nerel_bio.nerel_reader import create_instruct_dataset
        from utils.nerel_bio.nerel_bio_utils import ENTITY_TYPES

        test_path = os.path.join(arguments.data_path, 'test')
        test_dataset = create_instruct_dataset(test_path, max_instances=arguments.max_instances)
    else:
        from utils.conll2003.conll_reader import create_instruct_dataset
        from utils.conll2003.conll_utils import ENTITY_TYPES

        test_dataset = create_instruct_dataset(split='test', max_instances=arguments.max_instances)

    extracted_list = []
    target_list = []
    instruction_ids = []
    sources = []

    for instruction in tqdm(test_dataset):
        target_list.append(instruction['raw_entities'])
        instruction_ids.append(instruction['id'])
        sources.append(instruction['source'])

    target_list = list(batch(target_list, n=arguments.batch_size))
    instruction_ids = list(batch(instruction_ids, n=arguments.batch_size))
    sources = list(batch(sources, n=arguments.batch_size))

    for source in tqdm(sources):
        input_ids = tokenizer(source, return_tensors="pt", padding=True)["input_ids"].cuda()
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": 64,
        }
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generate_params,
                return_dict_in_generate=True,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )
        for s in generation_output.sequences:
            string_output = tokenizer.decode(s, skip_special_tokens=True)
            extracted_list.append(extract_classes(string_output, ENTITY_TYPES))

    pd.DataFrame({
        'id': np.concatenate(instruction_ids),
        'extracted': extracted_list,
        'target': np.concatenate(target_list)
    }).to_json(arguments.prediction_path)