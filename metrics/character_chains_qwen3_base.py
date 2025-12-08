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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm

# model_name="Qwen/Qwen3-8B"
# model_name="Qwen/Qwen3-32B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16  # V100: no bfloat16
)

def initialise_model(model_name: str, thinking: bool=False, bnb: bool=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if bnb:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )
    model.eval()

    # [151645, 151643]
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    tokenizer.padding_side = "left"

    return tokenizer, model, terminators

sys_prompt = ("""
You are an intelligent chatbot that can track characters in a list of audio descriptions.
Here's how you accomplish the task: you get a single audio description, and you will find character mentions (names, descriptions and pronouns).
You will provide the final result in JSON format. [{"text": mention-span, "label": ".." }, {"text": mention-span, "label": ".." }, ..].\n
Directly output the final sentence in the required JSON format. Do not include explanations or anything outside the JSON.
""")

sys_prompt_add_think = ("""
Your thinking must be brief, limit the chain to 4-5 sentences and you MUST end the thinking with a </think>.
After </think>, directly output the required JSON format. Do not include explanations or anything other than the JSON after </think>.
""")

user_prompt_de=("""
You will receive ONE audio-description sentence. Extract ONLY mentions of PEOPLE.
Return ONLY JSON: [{"text":..., "label":...}, ...] with labels in {NAME, PRN, DESC}.

Rules
- DESC = a noun phrase that itself denotes a person (head is human/agentive, e.g., "ein junger Mann", "die 30-Jährige", "die Schwarzhaarige").
  • Must be an NP referring to a person, not a property of a person.
  • Do NOT extract prepositional phrases (PPs) like "mit ..", "in ..", "ohne ..".
  • Do NOT extract clothing/body parts or lists of items (e.g., "Jeans", "Hemd und Wolljacke", "seine Haare").
- PRN = a pronoun referring to a person (er/sie/ihm/ihr, demonstratives like "Der"/"Dieser" when standing alone and human-referential).
  • Exclude possessives modifying non-person nouns ("sein Bart"), and pronouns referring to objects.
  • Exclude reflexive pronouns ("sich")
- NAME = proper names of people.

General
- Extract maximal person-denoting NP (don’t split into parts).
- Left-to-right, no duplicates. If none, return [].
- Output JSON only.

Examples (correct):
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

An der Wand hängen zwei Fotos.
[]

Der sieht sie an.
→ [{"text":"Der","label":"PRN"}, {"text":"sie","label":"PRN"}]

Examples (WRONG: exclude):
"in rotem Hemd"    # PP, not a person
"mit verschränkten Armen"  # PP, not a person
"Hemd und Wolljacke"       # clothing list
"seine braunen Haare"  # body part possessed
"sein/ihr/seine(m,n)/ihre(n,m)/sich" # possessive/reflexive pronouns

Another WRONG example:
Leute sitzen auf der Bank. Ein Becher kippt um. Er fällt zu Boden.
[{"text" : "Leute", "label": "DESC"}, {"text" : "Er", "label": "PRN"}]

"Er" in this context refers to the 'Becher', not to a person.
Here is the audio description for you to extract the mentions:

""")

user_prompt_it=("""
You will receive ONE audio-description sentence. Extract ONLY mentions of PEOPLE.
Return ONLY JSON: [{"text":..., "label":...}, ...] with labels in {NAME, PRN, DESC}.

Rules
- DESC = a noun phrase that itself denotes a person (head is human/agentive, e.g., "un uomo alto", "la donna di 30 anni", "la donna dai capelli neri").
  1. Must be an NP referring to a person, not a property of a person.
  2. Do NOT extract prepositional phrases (PPs) like "con ..", "in ..", "senza ..".
  3. Do NOT extract clothing/body parts or lists of items (e.g., "Jeans", "camicia e giacca di lana", "i suoi capelli").
- PRN = a pronoun referring to a PERSON
    1. A token can only be labeled PRN if it is a PRONOUN (not an article or determiner).
    2. Check if it refers to a PERSON (human individual).
        - PERSON pronouns include: lui, lei, loro, gli, le, lo, la, li, le, questo/quello when standing alone.
        - EXCLUDE possessives ("la sua barba"), reflexives ("si", "sé") and relative pronouns ("che")
        - if fused with a clitic, only extract the part that refers to a person: "glielo" -> "gli", "vederlo" -> "lo"
    3. If the pronoun refers to a THING or BODY PART, do NOT label it.
    4. If the token is used as an ARTICLE (preceding a noun), do NOT label it.
        - Example: "le mani", "la porta", "lo sguardo": EXCLUDE.
    5. Only label if the pronoun *stands alone* or replaces a person in context.
        - Example: "Lo guarda" (he looks at him): INCLUDE
        - "Lo sguardo", "gli occhi": EXCLUDE
- NAME = proper names of people.

General
- Extract maximal person-denoting NP (don’t split into parts).
- Left-to-right, no duplicates. If none, return [].
- Output JSON only.

Examples (correct):
Le donne scendono dall’autobus e prendono le valigie dal bagagliaio.
[{"text" : "Le donne", "label": "DESC"}]

Un uomo alto con un mazzo di fiori in mano osserva il flusso di donne.
[{"text" : "Un uomo alto", "label": "DESC"}]

Ha i capelli corti, biondo scuro, e la barba di tre giorni.
[]

Le porge il mazzo di fiori. Accanto a lei un bambino piccolo.
[{"text" : "Le", "label": "PRN"}, {"text": "lei", "label": "PRN"}, {"text": "un bambino piccolo", "label": "DESC"}]

Prende il libro dallo scaffale e glielo porge.
[{"text":"gli","label":"PRN"}]

Questo non sa cosa sta facendo.
[{"text":"Questo","label":"PRN"}]

Examples (WRONG: exclude):
"in camicia rossa"    # PP, not a person
"con le braccia conserte"  # PP, not a person
"i suoi capelli castani"  # body part possessed
"suo/sua/sé/si" # possessive/reflexive pronouns
"gli occhi, la porta, lo sguardo" # articles, not pronouns

Another WRONG example:
Riattacca il cellulare, lo posa sul tavolo.
[{"text" : lo", "label": "PRN"}]

"lo" in this context refers to the phone, not a person.
Here is the audio description for you to extract the mentions:

""")

def build_messages(sample: str, lang: str, thinking: bool=False):
    prompt = sys_prompt
    if thinking:
        prompt = sys_prompt + sys_prompt_add_think
    if lang == "de":
        return [{"role": "system", "content": prompt}, {"role": "user", "content": '\n'.join([user_prompt_de, sample + '\n'])}]
    elif lang == "it":
        return [{"role": "system", "content": prompt}, {"role": "user", "content": '\n'.join([user_prompt_it, sample + '\n'])}]
    else:
        print(f"Undefined language, can only use 'de' or 'it'.")
        exit(1)


def build_prompt(tokenizer: AutoTokenizer, sample: str, lang: str, thinking: bool=False):
    messages = build_messages(sample, lang, thinking)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # appends the assistant header so model knows to answer now
        enable_thinking=thinking
    )
    return prompt

def build_batched_prompts(tokenizer: AutoTokenizer, samples: list[str], lang: str, thinking: bool=False):
    return [build_prompt(tokenizer, s, lang, thinking) for s in samples]

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
                   terminators: list[int],
                   batch: list[str], # input strings, in case we need to rerun one without thinking
                   lang: str,
                   do_sample: bool=True,
                   temperature=0.7,
                   top_p: float=0.8,
                   top_k: float=20,
                   min_p: float=0,
                   thinking: bool=False):

    if thinking:
        max_new_tokens=4096
    else:
        max_new_tokens=256
    # print("max_new_tokens ", max_new_tokens)
    padded = tokenize_batch(tokenizer, prompts, model.device)

    gen_out = model.generate(
        **padded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=terminators,
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
        # print(f"thinking {thinking}")
        # print(f"output text: \n{text}")
        if thinking:
            if "</think>" in text: # split off the thinking part
                text = text.split('</think>')[-1].strip()
                # print("split text ", text)
            else:
                sample_text=batch[row_idx]
                print(f"no </think> found, trying sample without thinking: {sample_text}")
                nothink_prompt=build_batched_prompts(tokenizer, [sample_text], lang, thinking=False)
                nothink_padded = tokenize_batch(tokenizer, nothink_prompt, model.device)
                nothink_out = model.generate(
                    **nothink_padded,
                    max_new_tokens=256,
                    do_sample=do_sample,
                    eos_token_id=terminators,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    pad_token_id = tokenizer.pad_token_id
                )
                nothink_gen_only_ids = nothink_out[0][len(nothink_padded):]
                text = tokenizer.decode(nothink_gen_only_ids, skip_special_tokens=True)
                text = text.split('</think>')[-1].strip() # has an empty <think></think>
                print(f"no think text {text}")

        batch_texts.append(text.strip())
    return batch_texts

def repair_json_list(text):

    # 1) Count brackets
    open_sq = text.count('[')
    close_sq = text.count(']')
    open_br = text.count('{')
    close_br = text.count('}')

    print("Square brackets   [ : ] =", open_sq, close_sq)
    print("Curly braces      { : } =", open_br, close_br)

    # 2) Strip superfluous outer brackets (common LLM bug)
    # e.g. "[[ {...} ]]" → "[ {...} ]"
    # Keep stripping while the inside is still bracketed twice
    stripped = text.strip()
    while stripped.startswith('[[') and stripped.endswith(']]'):
        stripped = stripped[1:-1].strip()

    # 3) Ensure at least one outer list bracket pair
    # If not present, wrap text
    if not stripped.startswith('['):
        stripped = '[' + stripped
    if not stripped.endswith(']'):
        stripped = stripped + ']'

    # # 4) Fix curly braces by balancing
    # diff = open_br - close_br
    # if diff > 0:
    #     # Missing closing braces
    #     stripped += '}' * diff
    # elif diff < 0:
    #     # Too many closing braces
    #     # remove extras from the end
    #     for _ in range(-diff):
    #         stripped = stripped[::-1].replace('}', '', 1)[::-1]

    # 5) Final cleanup: remove commas before ]
    stripped = re.sub(r',\s*]', ']', stripped)
    return stripped


def check_json(outputs: list[str]):
    is_json = True

    for i,o in enumerate(outputs):
        # print(f"o: {o}")
        try:
            json.loads(o)
        except json.JSONDecodeError:
            # try to add outer brackets, missing sometimes
            print(f"Trying to repair json: {o}")
            o = repair_json_list(o)
            try:
                json.loads(o)
                outputs[i] =o
            except:
                print(f"Failed to parse repaired output: {o}")
                is_json = False
    return is_json, outputs

def main(args):


    # Read predicted output from Stage I
    df = pd.read_csv(args.AD_csv)

    # # Initialise the model
    tokenizer, model, terminators = initialise_model(args.model_name, args.thinking, args.bnb)
    batches = [df['text'].tolist()[i:i+args.batch_size] for i in range(0, len(df), args.batch_size)]

    mentions  = []
    for j, batch in enumerate(batches):
        print(f"batch {j}")
        batched_prompts = build_batched_prompts(tokenizer, batch, args.lang, args.thinking)
        batched_outputs = generate_batch(model, tokenizer, batched_prompts, terminators, batch, args.lang, thinking=args.thinking)

        # check if valid json
        is_json, outputs = check_json(batched_outputs)
        if not is_json:
            print(f"Not valid json: {outputs}")
            mentions.extend('[]') # add an empty list as placeholder
            exit(1)
            #continue

        mentions.extend(outputs)

    print(len(mentions))

    with open(args.out_csv, 'w') as o:
        for label in mentions:
            o.write(f"{label}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--AD_csv', default=None, type=str, help='csv with ADs')
    parser.add_argument('--out_csv', default=None, type=str, help='output csv')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B", help='model name, default: Qwen/Qwen3-8B')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--thinking', action='store_true', help='use thinking mode as default (will fall back to non-thinking if the model thinks a whole novel and runs out of tokens)')
    parser.add_argument('--bnb', action='store_true', help='use bitsandbytes to quantize (nf4)')
    parser.add_argument('--cache', type=str, metavar='PATH', help='transformers cache directory')
    parser.add_argument('--lang', type=str, help="data language (German or Italian)")
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    if args.cache:
        os.environ['HF_HOME'] = args.cache

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
