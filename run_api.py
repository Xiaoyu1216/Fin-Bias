import json
from datasets import Dataset
from tqdm import tqdm
import os
import pandas as pd
import argparse
from transformers import AutoTokenizer
import random
import logging
from typing import Union,Any
import asyncio
from tqdm.asyncio import tqdm_asyncio
import openai
from time import sleep
import aiolimiter
from openai import AsyncOpenAI, OpenAIError


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

def prepare_model_inputs(qa_data):
    model_inputs = []
    for example in tqdm(qa_data):
        system_input, user_input = prepare_cot_inv_input(example)
        model_input = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": user_input}
        ]       
        
        model_inputs.append(model_input)
    return model_inputs

async def _throttled_openai_chat_completion_acreate(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_completion_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    json_mode: bool = False,
):
    async with limiter:
        for _ in range(10):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    top_p=top_p,
                    response_format=None if not json_mode else { "type": "json_object" },
                )

            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                sleep(random.randint(5, 10))
        return None

async def generate_from_openai_chat_completion(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_completion_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
    json_mode: bool = False,
):
    """Generate from OpenAI Chat Completion API.

    Args:
        messages: List of messages to proceed.
        engine_name: Engine name to use, see https://platform.openai.com/docs/models
        temperature: Temperature to use.
        max_completion_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """    
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    
    limiter = aiolimiter.AsyncLimiter(requests_per_minute, 60)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            limiter=limiter,
            json_mode=json_mode,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if response:
            outputs.append(response.choices[0].message.content)
        else:
            outputs.append("Invalid Message")
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='gpt-5.1')
    
    # dataset and output
    parser.add_argument("--data_dir", type=str, default="analyst_report.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    # llm setting
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_completion_tokens", type=int, default=4096)
    
    
    # api key
    parser.add_argument("--requests_per_minute", type=int, default=100)
    
    args = parser.parse_args()
    df = pd.read_csv(args.data_dir)
    qa_data = Dataset.from_pandas(df)  
    
    suffix_model_name = args.model_name.split("/")[-1].replace(".", "_")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{suffix_model_name}.json")


    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        exit()

    model_inputs = prepare_model_inputs(qa_data)

    client = AsyncOpenAI(api_key="Put your own api key")

    raw_outputs = asyncio.run(generate_from_openai_chat_completion(client = client,
                                                    messages = model_inputs,
                                                    engine_name = args.model_name, 
                                                    temperature = args.temperature, 
                                                    top_p = args.top_p, 
                                                    max_completion_tokens = args.max_completion_tokens,
                                                    requests_per_minute = args.requests_per_minute,))
    
    
        
        
    
        # Adds the model’s response to the original example.
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