import sys
import json
import fire

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from bencao_util import Prompter

if torch.cuda.is_available():
    device = "cuda"

def load_instruction(instruct_dir):
    input_data = []
    with open(instruct_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            d = json.loads(line)
            input_data.append(d)
    return input_data

import json
def data(path):
    sentences = []
    labels = []
    with open(path,'r',encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            content = data
            sentence = content[8] + content[10] + content[11]
            # sentence = sentence.replace('，', '').replace('。', '').replace('、', '').replace('；', '').replace('：',
            #                                                                                                 '').replace(
            #     '？', '').replace('！', '').replace('\t', '').replace('主诉', '').replace('现病史', '').replace('*',
            #                                                                                                   '').replace(
            #     ' ', '').replace('“', '').replace('表格<诊断>内容', '').replace('\n', '')
            sentences.append(sentence[0:500])
            labels.append(content[-1].split('|'))
    return sentences, labels

def main(
    load_8bit: bool = False,
    base_model: str = "./bencao",
    instruct_dir: str = "",
    use_lora: bool = True,
    lora_weights: str = "./bencao/lora-bloom-med-bloom",
    # The prompt template to use, will default to med_template.
    prompt_template: str = "med_template",
):
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_lora:
        print(f"using lora {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=1,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)


    syndrome = 'A：肝阳上亢证   B:痰湿痹阻证   C:气滞血瘀证   D:湿热蕴结证   E:痰瘀互结证   F:阴虚阳亢证   G:气虚血瘀证  H：痰热蕴结证  I:心气虚血瘀证  J：气阴两虚证'
    syndrome_id = {'肝阳上亢证': 'A', '痰湿痹阻证': 'B', '气滞血瘀证': 'C', '湿热蕴结证': 'D', '痰瘀互结证': 'E',
                   '阴虚阳亢证': 'F', '气虚血瘀证': 'G', '痰热蕴结证': 'H', '心气虚血瘀证': 'I', '气阴两虚证': 'J'}
    sentences, labels = data('./7分类/dev.json')
    for sentence in sentences:
        # now
        # query = 'This is a multiple choice question. \n' \
        #         'The patients condition is described as: {} \n' \
        #         'Based on the above patients condition description, output the patient choice of the most appropriate option among the seven syndrome types {}: \n'.format(sentence, syndrome)
        query = '患者的中医四诊信息为：{} \n' \
                '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
            sentence, syndrome)

        # print("Instruction:", query)
        # xx = evaluate(query)
        print("Response:", evaluate(query))
        print()


if __name__ == "__main__":
    fire.Fire(main)