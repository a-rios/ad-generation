import json
import pandas as pd
import argparse
import regex as re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import ast



def get_user_prompt(sentences, lang):

    examples = ""
    for inp, out in few_shot_examples[lang]:
        examples += f"Input: {inp}\nOutput: {out}\n\n"

    inputs = '\n'.join([s for s in sentences])

    user_prompt = (
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\nIn the following {lang} sentences, replace person names with the corresponding pronouns. Also replace names in possessive phrase with the corresponding possessive pronoun. Examples:\n\n{examples}\nNow rephrase the following sentences:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{inputs}"
        )
    return user_prompt

def chunk_sentences(sentences, batch_size):
    """Chunk a list of sentences into batches."""
    for i in range(0, len(sentences), batch_size):
        yield sentences[i:i + batch_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_jsons', metavar='PATH', nargs='+', type=str, help="input json with ADs and start/end times")
    parser.add_argument('--out_csv', metavar='PATH',type=str, help="output csv with original ground truth AD + AD with names replaced by pronouns")
    parser.add_argument('--lang', type=str, help="language")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--cache_dir', type=str, metavar='PATH', help="huggingface cache dir")
    parser.add_argument('--llama_key', type=str,  help="access token")

    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, token=args.llama_key)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", cache_dir=args.cache_dir, token=args.llama_key)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    sys_prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are an intelligent chatbot that assists with rephrasing audio descriptions to remove explicit character information. "
                "Here's how you can accomplish the task: Rephrase the provided audio descriptions to remove character names. If there is no character name in the sentence, repeat the sentence unchanged. Do not change character references by looks or features, only replace names."
                "You should directly start the answer with the converted results WITHOUT providing ANY more sentences at the beginning or at the end."
                "Format your output as tuples of input and output:\n"
                "('input1', 'output1')"
                "('input2', 'output2')"
        )

    few_shot_examples = {
            "German": [
                ("Anna Meier öffnet die Tür und schaut hinaus.", "Sie öffnet die Tür und schaut hinaus."),
                ("Peter's Schirm ist blau.", "Sein Schirm ist blau."),
                ("Maria sieht Benjamin mit dem Koffer auf dem Bahnsteig.", "Sie sieht ihn mit dem Koffer auf dem Bahnsteig."),
                ("Hans auf dem Dach.", "Er ist auf dem Dach."),
                ("Hans und Peter sprechen miteinenander.", "Sie sprechen miteinenander."),
                ("Die Rothaarige steht vor dem Haus.", "Die Rothaarige steht vor dem Haus."),
            ],
            "Italian": [
                ("Anna apre la porta e guarda fuori.", "Lei apre la porta e guarda fuori."),
                ("Marco prende il suo telefono.", "Lui prende il suo telefono."),
                ("Il libro di Lucia è sul tavolo.", "Il suo libro è sul tavolo."),
                ("Gianni Bianchi sul tetto.", "Lui è sul tetto."),
                ("Peter e Hans stanno pescando.", "Stanno pescando.")
            ],
            "French": [
                ("Claire ouvre la porte et regarde dehors.", "Elle ouvre la porte et regarde dehors."),
                ("Paul Dubois prend son téléphone.", "Il prend son téléphone."),
                ("Le sac de Julie est rouge.", "Son sac est rouge."),
                ("Jean sur le toit.", "Il est sur le toit."),
                ("Peter et Hans pêchent.", "Ils pêchent.")
            ]
    }

    samples = []
    out_samples = []

    for in_file in args.in_jsons:
        with open(in_file, 'r') as f:
            data = json.load(f)
            samples.extend([s['source'].strip() for s in data if not re.match(r'(\$|UT|Untertitel:)', s['source'])])

    print("samples , ", len(samples))

    for batch in chunk_sentences(samples, args.batch_size):
            # print(batch)
            user_prompt=get_user_prompt(batch, args.lang)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipe(
                messages,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id = tokenizer.eos_token_id,
            )

            output_text = outputs[0]["generated_text"][2]['content'] #[{'role': 'system', 'content': string}, {'role': 'user', 'content': string}, {'role': 'assistant', 'content': string}]
            # sometimes we get a comma separated list, other times just tuples with \n
            output_as_list = f"[{output_text.replace('\n', ',').replace(',,', ',') }]"
            try:
                out = ast.literal_eval(output_as_list)
                # check that all tuples are actually tuples
                if all(isinstance(item, tuple) for item in out):
                    out_samples.extend(out)
                else:
                    print(f"List contains invalid tuple(s): {out}")
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing: \n{output_as_list}\n {e}")
                continue

    df = pd.DataFrame(out_samples, columns=["text_gt", "text_gt_wo_char"])
    df.to_csv(args.out_csv, index=False)
    #print(df.to_string(index=False))


