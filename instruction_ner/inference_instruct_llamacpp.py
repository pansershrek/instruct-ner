import re
import os
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig
#from llama_cpp import Llama
from utils.rudrec.rudrec_reader import create_train_test_instruct_datasets
from utils.nerel_bio.nerel_reader import create_instruct_dataset
from utils.conll2003.conll_reader import create_instruct_dataset as create_instruct_dataset_conll
from metric import extract_classes
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForSeq2Seq, GenerationConfig,
    DataCollatorForTokenClassification, EvalPrediction,
    T5ForConditionalGeneration, Trainer, TrainerCallback,
    TrainerControl, TrainerState, TrainingArguments
)
from train_utils import fix_model, fix_tokenizer, set_random_seed, SUPPORTED_DATASETS
from peft import (
    LoraConfig, PeftConfig, PeftModel, get_peft_model,
    prepare_model_for_kbit_training
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='conll2003', type=str, help='name of dataset')
    parser.add_argument("--data_path", default='data/rudrec/rudrec_annotated.json', type=str, help='train file_path')
    parser.add_argument("--model_path", default='poteminr/llama2-rudrec', type=str, help='ggml model path')
    parser.add_argument("--model_name", default="bigscience/bloomz-3b", type=str, help='model name from hf')
    parser.add_argument("--prediction_path", default='prediction.json', type=str, help='path for saving prediction')
    parser.add_argument("--max_instances", default=-1, type=int, help='max number of instruction')
    parser.add_argument("--max_new_tokens", default=512, type=int, help='max number of generated tokens')
    arguments = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        arguments.model_name,
        #load_in_8bit=True,
        device_map='cpu',
        #n_gpu_layers = 35,
        #n_ctx=2048,
        #n_parts=1,
        #use_mmap=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(arguments.model_name)
    model = fix_model(model, tokenizer, use_resize=False)
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(
        model,
        "/home/admin/instruct-ner/instruction_ner/models/checkpoint-219",
    )
    model = model.merge_and_unload()
    max_new_tokens = arguments.max_new_tokens

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

    if arguments.dataset_name == 'rudrec':
        from utils.rudrec.rudrec_utis import ENTITY_TYPES
        _, test_dataset = create_train_test_instruct_datasets(arguments.data_path)
        if arguments.max_instances != -1 and arguments.max_instances < len(test_dataset):
            test_dataset = test_dataset[:arguments.max_instances]
    elif arguments.dataset_name == 'nerel_bio':
        from utils.nerel_bio.nerel_bio_utils import ENTITY_TYPES
        test_path = os.path.join(arguments.data_path, 'test')
        test_dataset = create_instruct_dataset(test_path, max_instances=arguments.max_instances)
    else:
        from utils.conll2003.conll_utils import ENTITY_TYPES
        test_dataset = create_instruct_dataset_conll(
            split='validation', max_instances=arguments.max_instances
        )

    extracted_list = []
    target_list = []
    instruction_ids = []
    sources = []

    for instruction in tqdm(test_dataset):
        #input_ids = model.tokenize(instruction['source'])
        input_ids = tokenizer.encode(
            instruction['source']
        )
        input_ids.append(tokenizer.eos_token_id)
        generator = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                reset=True,
        )

        completion_tokens = []
        for i, token in enumerate(generator):
            completion_tokens.append(token)
            if token == tokenizer.token_eos or (max_new_tokens is not None and i >= max_new_tokens):
                break

        completion_tokens = model.detokenize(completion_tokens).decode("utf-8")
        extracted_list.append(extract_classes(completion_tokens), ENTITY_TYPES)
        instruction_ids.append(instruction['id'])
        target_list.append(instruction['raw_entities'])

    pd.DataFrame({
        'id': instruction_ids,
        'extracted': extracted_list,
        'target': target_list
    }).to_json(arguments.prediction_path)