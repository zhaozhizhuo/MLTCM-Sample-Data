from openai import OpenAI
import httpx

client = OpenAI(
    base_url="url",
    api_key="api-key",
    http_client=httpx.Client(
        base_url="url",
        follow_redirects=True,
    ),
)

import json
def data(path):
    sentences = []
    labels = []
    with open(path,'r',encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            content = data
            sentence = content[10] + content[16]
            sentences.append(sentence[0:500])
            labels.append(content[-1].split('|'))
    return sentences, labels


letters = [i for i in range(1,200)]
syndrome_id = {}
with open('./cardiovascular.json','r',encoding='utf-8') as file:
    for i,line in enumerate(file):
        syndrome_id[letters[i]] = line.replace('\n','')

syndrome = '  '.join([f"{key}: {value}" for key, value in syndrome_id.items()])

sentences, labels = data('./dev.json')


pre_lable = []
for sentence in sentences:
  query = 'This is a multiple choice TCM syndrome differentiation task. \n' \
          'only need to output the corresponding options, not explanation! You need to select one syndrome suitable for the patient among the ten options ({}): \n'\
          'The patients four diagnosis information is described as: {} \n' \
                .format(syndrome,sentence)
  # print(query)
  completion = client.chat.completions.create(
    max_tokens=100,
    model="gpt-3.5-turbo",
    messages=[
        # {"role": "system", "content": "This is a multiple choice TCM syndrome differentiation task \n' \
        #   'You need to select one diagnosis types suitable for the patient among the ten options , and only need to output the corresponding options: \n'\
        #   'The patients four diagnosis information is described as: {} \n' ".format('A')},
        {"role": "user", "content": "{}".format(query)}
    ]
  )
  pre_lable.append(completion.choices[0].message.content)
  print(completion.choices[0].message.content)

with open('./openai_dev.json', 'w', encoding='utf-8') as file:
  for i in range(len(pre_lable)):
    file.write(json.dumps({'id': i, 'label': pre_lable[i]}, ensure_ascii=False) + '\n')