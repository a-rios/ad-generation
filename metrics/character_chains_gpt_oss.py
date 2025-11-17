import os
import sys
import ast
import re
import json
import torch
import random
import argparse
import numpy as np
# import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

## needs an A100 or H100, Mxfp4
model_name="openai/gpt-oss-20b"

def initialise_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )
    model.eval()

    tokenizer.padding_side = "left"

    return tokenizer, model

sys_prompt = ("""
You are an intelligent chatbot that can track characters in a list of audio descriptions.
Here's how you accomplish the task: you get a single audio description, and you will find character mentions (names, descriptions and pronouns).
You will provide the final result in JSON format. [{"text": mention-span, "label": ".." }, {"text": mention-span, "label": ".." }, ..].\n
Directly output the final sentence in the required JSON format. Do not include explanations or anything outside the JSON.
""")


user_prompt = ("""
You will receive a single audio description. You will extract character mentions: names, pronouns and descriptions of their appearance. Provide a JSON list of dictionaries with the mentions. Each mention has a "text" attribute with the text span, and a "label" attribute with the corresponding label. Labels are: DESC (=description of the character), NAME (= character name), PRN (= pronoun). If there is no character mention in the sentence, return an empty list: []. Do not include anything but the JSON in the reply.
Here are some examples with inputs and outputs:

Die Frauen steigen aus dem Bus und holen Koffer aus dem Gepäckfach.
[{"text" : "Die Frauen", "label": "DESC"}]

Ein junger Mann mit einem Blumenstrauss in der Hand sucht den Strom von Frauen ab.
[{"text" : "Ein junger Mann", "label": "DESC"}]

Er hat kurze, dunkelblonde Haare und einen Dreitagebart.
[{"text" : "Er", "label": "PRN"}]

Der Bus fährt ab.
[]

Er streckt ihr den Blumenstrauss entgegen. Neben ihr ein kleines Kind.
[{"text" : "Er", "label": "PRN"}, {"text": "ihr", "label": "PRN"}, {"text": "ihr", "label" : "PRN"}, {"text": "ein kleines Kind", "label": "DESC"}]

Sophie schüttelt den Kopf.
[{"text" : "Sophie", "label": "NAME"}]

An der Wand hängen zwei Fotos.
[]

Elsa geht zögernd einen Schritt auf sie zu.
[{"text" : "Elsa", "label": "NAME"}, {"text": "sie", "label": "PRN"}]

Here is the audio description:
"""
)

def build_messages(sample):
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": '\n'.join([user_prompt, sample + '\n'])}]


def build_prompt(tokenizer: AutoTokenizer, sample: str):
    messages = build_messages(sample)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # appends the assistant header so model knows to answer now
    )
    return prompt

def build_batched_prompts(tokenizer: AutoTokenizer, samples: list[str]):
    return [build_prompt(tokenizer, s) for s in samples]

def tokenize_batch(tokenizer: AutoTokenizer, prompts: list[str], device: torch.device):

    padded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)

    return padded


@torch.no_grad()
def generate_batch(model: AutoModelForCausalLM,
                   tokenizer: AutoTokenizer,
                   prompts: list[str],
                   do_sample: bool=True,
                   temperature=0.7,
                   top_p: float=0.8,
                   top_k: float=20,
                   min_p: float=0):

    max_new_tokens=4096
    padded = tokenize_batch(tokenizer, prompts, model.device)

    gen_out = model.generate(
        **padded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        pad_token_id = tokenizer.pad_token_id
    )

    batch_texts = []
    padded_length = len(padded[0]) #  padding is left, so length of padded = padding + prompt
    for row_idx, full_ids in enumerate(gen_out):
        gen_only_ids = full_ids[padded_length:] # slice off prompt + padding

        text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
        if "assistantfinal" in text: # split off the thinking part
                text = text.split('assistantfinal')[-1].strip()
        else:
            ## without thinking: add <|channel|>final<|message|> = 200005, 17196, 200008 to prompts (https://huggingface.co/openai/gpt-oss-120b/discussions/50)
            input_ids = padded['input_ids'][row_idx]
            extra = torch.tensor([200005, 17196, 200008], device=input_ids.device)
            extra = extra.unsqueeze(0).expand(input_ids.size(0), -1)  # (batch, 3)
            input_ids = torch.cat([input_ids, extra], dim=1)
            print("new inputs ", input_ids)

            att_mask = padded['attention_mask'][row_idx]
            att_mask = torch.cat([att_mask, torch.ones((input_ids.size(0), 3), dtype=att_mask.dtype, device=input_ids.device)], dim=1)

            row_out = model.generate(
                input_ids=input_ids,
                attention_mask=att_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                pad_token_id = tokenizer.pad_token_id
            )
            gen_only_ids = full_ids[padded_length:]
            text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
            print("second try text ", text)
            text = text.split('assistantfinal')[-1].strip()

        print(f"gen text: {text}")

        batch_texts.append(text.strip())
    return batch_texts


def check_json(outputs: list[str]):
    is_json = True

    for i,o in enumerate(outputs):
        # print(f"o: {o}")
        try:
            json.loads(o)
        except json.JSONDecodeError:
            # try to add outer brackets, missing sometimes
            o = f"[{o}]"
            print(f"Adding brackets: {o}")
            try:
                json.loads(o)
                outputs[i] =o
            except:
                print(f"Failed to parse output: {o}")
                is_json = False
    return is_json, outputs

def main(args):


    # Read predicted output from Stage I
    df = pd.read_csv(args.AD_csv)

    # # Initialise the model
    tokenizer, model = initialise_model()
    batches = [df['text'].tolist()[i:i+args.batch_size] for i in range(0, len(df), args.batch_size)]

    mentions  = []
    for batch in batches:
        batched_prompts = build_batched_prompts(tokenizer=tokenizer, samples=batch)
        batched_outputs = generate_batch(model=model, tokenizer=tokenizer, prompts=batched_prompts)


        # check if valid json
        is_json, outputs = check_json(batched_outputs)
        if not is_json:
            print(f"Not valid json: {outputs}")
            exit(1)

        mentions.extend(outputs)

    print(len(mentions))

    with open(args.out_csv, 'w') as o:
        for label in mentions:
            o.write(f"{label}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--AD_csv', default=None, type=str, help='csv with ADs')
    parser.add_argument('--out_csv', default=None, type=str, help='output csv')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--cache', type=str, metavar='PATH', help='transformers cache directory')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    if args.cache:
        os.environ['HF_HOME'] = args.cache

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
