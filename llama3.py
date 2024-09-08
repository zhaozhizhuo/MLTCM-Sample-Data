from transformers import AutoTokenizer, AutoConfig, AddedToken, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dataclasses import dataclass
from typing import Dict
import torch
import copy
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

## 定义聊天模板
@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str

template_dict: Dict[str, Template] = dict()

def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
    )

# 这里的系统提示词是训练时使用的，推理时可以自行尝试修改效果
register_template(
    template_name='llama3',
    system_format='<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|end_of_text|>\n',
    system="You are a helpful, excellent and smart assistant. "
        "Please respond to the user using the language they input, ensuring the language is elegant and fluent."
        "If you don't know the answer to a question, please don't share false information.",
    stop_word='<|end_of_text|>'
)


## 加载模型
def load_model(model_name_or_path, load_in_4bit=False, adapter_name_or_path=None):
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    # 加载base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=quantization_config
    )

    # 加载adapter
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)

    return model

## 加载tokenizer
def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

## 构建prompt
def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    history.append({"role": '你是一个中医辨证专家', 'message': query})
    input_ids = []

    # 添加系统信息
    if system_format is not None:
        if system is not None:
            system_text = system_format.format(content=system)
            input_ids = tokenizer.encode(system_text, add_special_tokens=False)
    # 拼接历史对话
    for item in history:
        role, message = item['role'], item['message']
        if role == '你是一个中医辨证专家':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
        tokens = tokenizer.encode(message, add_special_tokens=False)
        input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


import json
from tqdm import tqdm
def data(path):
    sentences = []
    labels = []
    with open(path,'r',encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            content = data
            # if args.syndrome_diag == 'syndrome':
            sentence = content[10] + content[16]
            # if args.syndrome_diag == 'diag':
            # sentence = content[9] + '[SEP]' + content[11] + '[SEP]' + content[13]
            sentences.append(sentence[0:500])
            labels.append(content[-1].split('|'))
    return sentences, labels

def main():

    #加载数据和和历史记录
    letters = [i for i in range(1, 147)]
    syndrome_id = {}
    with open('./7分类/pure/cardiovascular.json', 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            syndrome_id[letters[i]] = line.replace('\n', '')

    syndrome = '  '.join([f"{key}: {value}" for key, value in syndrome_id.items()])
    syndrome_id = {value: key for key, value in syndrome_id.items()}

    sentences_test, labels = data('./pure/test.json')
    sentences_dev, labels = data('./pure/dev.json')
    his_sen, his_labels = data('./history.json')

    # 历史
    history = []
    for i in range(len(his_sen)):
        history_i ={}
        query = '你需要依据患者的中医四诊信息判断证型，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
                '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
            his_sen[i], syndrome)
        ids = []
        for his_label in his_labels[i]:
            id = syndrome_id[his_label]
            ids.append(str(id))
        # ans = ' '.join(ids)
        ans = '；'.join(his_labels[i])
        history_i['role'] = query
        history_i['message'] = ans
        history.append(history_i)


    model_name_or_path = '/zy_data_encoder/llama3/Llama3-Chinese_v2/' # 模型名称或路径，请修改这里
    template_name = 'llama3'
    adapter_name_or_path = None

    template = template_dict[template_name]
    # 若开启4bit推理能够节省很多显存，但效果可能下降
    load_in_4bit = False

    # 生成超参配置，可修改以取得更好的效果
    max_new_tokens = 100 # 每次回复时，AI生成文本的最大长度
    top_p = 0.9
    temperature = 0.6 # 越大越有创造性，越小越保守
    repetition_penalty = 1.1 # 越大越能避免吐字重复

    # 加载模型
    print(f'Loading model from: {model_name_or_path}')
    print(f'adapter_name_or_path: {adapter_name_or_path}')
    model = load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
    if template.stop_word is None:
        template.stop_word = tokenizer.eos_token
    stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=True)
    assert len(stop_token_id) == 1
    stop_token_id = stop_token_id[0]


    pre_label = []
    for sentence in tqdm(sentences_dev):
        query = '你需要依据患者的中医四诊信息判断证型，只需要告诉我证型对应的数字即可，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
                '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
            sentence, syndrome)
        # query = query.strip()
        input_ids = build_prompt(tokenizer, template, query, copy.deepcopy(history), system=None).to(model.device)
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=stop_token_id
        )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(template.stop_word, "").strip()

        # # 存储对话历史
        # history.append({"role": 'user', 'message': query})
        # history.append({"role": 'assistant', 'message': response})

        # 当对话长度超过6轮时，清空最早的对话，可自行修改
        if len(history) > 12:
            history = history[:-12]

        response = response.split('\n\n')[-1]
        print("# Llama3-Chinese：{}".format(response))
        pre_label.append(response)

    with open('./llama3-dev_pure.json','w',encoding='utf-8') as file:
        for i in range(len(pre_label)):
            file.write(json.dumps({'id': i, 'label': pre_label[i]}, ensure_ascii=False) + '\n')

    pre_label = []
    for sentence in tqdm(sentences_test):
        query = '你需要依据患者的中医四诊信息判断证型，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
                '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
            sentence, syndrome)
        # query = query.strip()
        input_ids = build_prompt(tokenizer, template, query, copy.deepcopy(history), system=None).to(model.device)
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=stop_token_id
        )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(template.stop_word, "").strip()

        # # 存储对话历史
        # history.append({"role": 'user', 'message': query})
        # history.append({"role": 'assistant', 'message': response})

        # 当对话长度超过6轮时，清空最早的对话，可自行修改
        if len(history) > 12:
            history = history[:-12]

        response = response.split('\n\n')[-1]
        print("# Llama3-Chinese：{}".format(response))
        pre_label.append(response)

    with open('./llama3-test_pure.json', 'w', encoding='utf-8') as file:
        for i in range(len(pre_label)):
            file.write(json.dumps({'id': i, 'label': pre_label[i]}, ensure_ascii=False) + '\n')



if __name__ == '__main__':
    main()