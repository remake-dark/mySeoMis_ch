import torch
import random
import pandas as pd
import json
import jieba
import numpy as np
import nltk
import tqdm
from transformers import BertTokenizer
from transformers import RobertaTokenizer, T5Tokenizer, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

label_dict = {
    "real": 0,
    "fake": 1
}

domain_dict = {
    "社会生活": 0,
    "无法确定": 0,
    "军事": 1,
    "科技": 1,
    "文体娱乐": 2,
    "财经商业": 3,
    "政治": 4,
    "医药健康": 5,
    "灾难事故": 6,
    "教育考试": 7
}

def extract_entities(data):
    def parse_entities(text):
        last_sep_index = text.rfind('。[SEP]')
        if last_sep_index != -1:
            entities_part = text[last_sep_index + len('。[SEP]'):]
        else:
            entities_part = text
        entities = entities_part.split('[SEP]')

        entities = [entity for entity in entities if entity.strip()]
        
        return entities

    if isinstance(data, list):
        return [parse_entities(item) for item in data]
    else:
        return parse_entities(data)


def word2input_ro(texts, max_len):
    tokenizer = BertTokenizer.from_pretrained('chinese-roberta-wwm-ext')
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def word2input_t5(texts, max_len):
    tokenizer =  AutoTokenizer.from_pretrained('/mySeoSent/mySeoSent_ch/mySeoSent_ch/mt5/damo/nlp_mt5_zero-shot-augment_chinese-base', do_lower_case=False)
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def word2input(texts, max_len):
    tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def data_augment(content, event_content, aug_prob):
    entity_list = extract_entities(event_content)
    entity_content = []
    random_num = random.randint(1,100)
    if random_num <= 50:
        for item in entity_list:
            random_num = random.randint(1,100)
            if random_num <= int(aug_prob * 100):
                content = content.replace(item, '[MASK]')
                event_content = event_content.replace(item, '[MASK]')
            elif random_num <= int(2 * aug_prob * 100):
                content = content.replace(item, '')
                event_content = event_content.replace(item, '')
            else:
                a=1
                #entity_content.append(item["entity"])
        #entity_content = ' [SEP] '.join(entity_content)
    else:
        content = list(jieba.cut(content))
        #content = list(nltk.word_tokenize(content))
        for index in range(len(content) - 1, -1, -1):
            random_num = random.randint(1,100)
            if random_num <= int(aug_prob * 100):
                del content[index]
            elif random_num <= int(2 * aug_prob * 100):
                content[index] = '[MASK]'
        content = ''.join(content)

        event_content = list(jieba.cut(event_content))
        #event_content = list(nltk.word_tokenize(event_content))
        for index in range(len(event_content) - 1, -1, -1):
            random_num = random.randint(1,100)
            if random_num <= int(aug_prob * 100):
                del event_content[index]
            elif random_num <= int(2 * aug_prob * 100):
                event_content[index] = '[MASK]'
        event_content = ''.join(event_content)

    return content, event_content

def get_entity(event_content):
    #entity_content = []
    #for item in entity_list:
        #entity_content.append(item["entity"])
    #entity_content = ' [SEP] '.join(entity_content)
    return event_content

def get_dataloader(path, max_len, batch_size, shuffle, use_endef, aug_prob, tokenizer_type = 'bert', soc_feat_path = './data/soc_feat_w_bert_train.npy'):
    if "val" in path.lower():
        soc_feat_path = './data/soc_feat_w_bert_val.npy'
        path = './data/val.json'
    elif "test" in path.lower():
        soc_feat_path = './data/soc_feat_w_bert_test.npy'
        path = './data/test.json' 
    else:
        path = './data/train.json'
    with open(path, 'r', encoding='utf-8') as file:
        data_list = [json.loads(line) for line in file]
    #data_list = json.load(open(path, 'r',encoding='utf-8'))
    df_data = pd.DataFrame(columns=('content','label', 'event', 'domain'))
    for item in tqdm(data_list, desc="Dataloader: loading items from current dataset"):
        tmp_data = {}
        if shuffle == True and use_endef == True:
            tmp_data['content'], tmp_data['event'] = data_augment(item['content'], item['entity_list'][0], aug_prob)
        else:
            tmp_data['content'] = item['content']
            tmp_data['event'] =  item['entity_list'][0]                     #get_entity(item['entity_list'])
        tmp_data['label'] = item['label']
        tmp_data['domain'] = item['domain']
        #tmp_data['year'] = item['time'].split(' ')[0].split('-')[0]
        #df_data = df_data.append(tmp_data, ignore_index=True)
        
        df_data = pd.concat([df_data, pd.DataFrame([tmp_data])], ignore_index=True)

    #emotion = np.load(path.replace('.json', '_emo.npy')).astype('float32')
    #emotion = torch.tensor(emotion)
    content = df_data['content'].to_numpy()
    event_content = df_data['event'].to_numpy()
    label = torch.tensor(df_data['label'].astype(int).to_numpy())
    domain = torch.tensor(df_data['domain'].apply(lambda c: domain_dict[c]).astype(int).to_numpy())

    if 'roberta' in tokenizer_type.lower():
        content_token_ids, content_masks = word2input_ro(content, max_len)
        event_token_ids, event_masks = word2input_ro(event_content, 100)
    elif 't5' in tokenizer_type.lower():
        content_token_ids, content_masks = word2input_t5(content, max_len)
        event_token_ids, event_masks = word2input_t5(event_content, 100)
    else:
        content_token_ids, content_masks = word2input(content, max_len)
        event_token_ids, event_masks = word2input(event_content, 100)  #改一下

    soc_emb_dim = 768
    kernel_dim = 22
    soc_feat_np = np.load(soc_feat_path)
    soc_feat_2D = torch.from_numpy(soc_feat_np)
    soc_feat_cens = soc_feat_2D[:, (-2)*soc_emb_dim:]
    pos_soc_feat_cens = soc_feat_cens[:, 0:soc_emb_dim].unsqueeze(1)
    neg_soc_feat_cens = soc_feat_cens[:, soc_emb_dim: 2*soc_emb_dim].unsqueeze(1)
    soc_feat = torch.cat((pos_soc_feat_cens, neg_soc_feat_cens), dim = 1)
    kernel_features = soc_feat_2D[:, 0: kernel_dim*2]
    
    soc_feat_masks = torch.ones((soc_feat_2D.shape[0], 2), dtype=torch.long)
    dataset = TensorDataset(content_token_ids,
                            content_masks,
                            event_token_ids,
                            event_masks,
                            label,
                            domain,
                            kernel_features,
                            soc_feat,
                            soc_feat_masks
                            )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader