from vllm.entrypoints.llm import LLM # from vllm import LLM (recommended)
from vllm.sampling_params import SamplingParams
from tqdm import tqdm
import json
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import os
import argparse
from transformers import AutoTokenizer
import random
from typing import Union
from utils import *


def process_single_example_raw_outputs(outputs):
    processed_outputs = []
    assert len(outputs.outputs) == 1
    processed_outputs.append(outputs.outputs[0].text)
    return processed_outputs

def prepare_cot_inv_input(example):
    system_input = (
        "You are an investor. Analyze the firm analyst report logically.\n"
        "Then provide your own investment rating.\n"
        "Format as a JSON object with the following fields:\n"
        "answer: The precise answer to the question. Only one of {bullish, neutral, bearish}.\n"
        "reason: One or more paragraphs indicating why you provide the answer."
    )

    user_input = example['clean_content']

    return system_input, user_input

def prepare_model_inputs(qa_data, prompt_type, model_name, tokenizer=None):
    model_inputs = []
    for example in tqdm(qa_data):
        system_input, user_input = prepare_cot_inv_input(example)
        
        models_without_system = ("gemma", "OLMo", "Mistral", "Mixtral", "starcoder2")
        if any(model in model_name for model in models_without_system):
            model_input = [
                {"role": "user", "content": system_input + "\n" + user_input}
            ]
        else:
            model_input = [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input}
            ]
        
            # .apply_chat_template(...) takes the structured list of messages (with roles) and turns it into a text string in the format the model expects.
        model_input = tokenizer.apply_chat_template(model_input, tokenize=False)
        
        model_inputs.append(model_input)
    return model_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    
    # dataset and output
    parser.add_argument("--data_dir", type=str, default="analyst_report.csv")
    
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    # llm setting
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=int, default=1.0)
    parser.add_argument("--prompt_type", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_num_examples", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--quantization", type=str, default="")
    
    
    args = parser.parse_args()
    # Detects how many GPUs are available by reading the CUDA_VISIBLE_DEVICES environment variable.
    gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    df = pd.read_csv(args.data_dir)
    qa_data = Dataset.from_pandas(df)
    
    if args.max_num_examples > 0:
        random.shuffle(qa_data)
        qa_data = qa_data[:args.max_num_examples]
    
    suffix_model_name = args.model_name.split("/")[-1].replace(".", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, args.subset, f"raw_{args.prompt_type}_outputs")
    # If output_dir does not exist, it will be created (along with any missing parent directories).
    # If output_dir already exists, nothing happens — Python silently skips folder creation.

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{suffix_model_name}.json")


    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        exit()

    
    if args.quantization:
        llm = LLM(args.model_name,
                tensor_parallel_size=gpu_count,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=True,
                quantization=args.quantization)
    else:
        llm = LLM(args.model_name, 
                tensor_parallel_size=gpu_count, 
                dtype="half" if "gemma-2" not in args.model_name else "bfloat16", # https://github.com/vllm-project/vllm/issues/6177
                swap_space=16, 
                gpu_memory_utilization=args.gpu_memory_utilization, 
                trust_remote_code=True)
    
    sampling_params = SamplingParams(temperature = args.temperature, 
                                    top_p = args.top_p, 
                                    max_tokens = args.max_tokens)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, verbose=False, trust_remote_code=True)
    # When you're explicitly providing a system message, the tokenizer won’t inject another one — 
    # use_default_system_prompt only applies when no system message is present in the message list.
    tokenizer.use_default_system_prompt = True
    model_inputs = prepare_model_inputs(qa_data, args.prompt_type, args.model_name, tokenizer) 
    
    outputs = llm.generate(model_inputs, sampling_params)
    raw_outputs = [process_single_example_raw_outputs(output) for output in outputs]
    
    
    output_data = []
    for raw_output, qa in zip(raw_outputs, qa_data):
        if type(raw_output) != list:
            qa["output"] = [raw_output]
        else:
            qa["output"] = raw_output
        output_data.append(qa)
    # indent = 4: Makes the output JSON file pretty-printed with 4-space indentation.
    # Ensures that all characters in the output JSON file are escaped into ASCII.
    json.dump(output_data, open(output_file, "w"), indent=4, ensure_ascii=True)