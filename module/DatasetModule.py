import torch
from fastchat.model.model_adapter import get_conversation_template
from torch.utils.data import Dataset
from transformers import LlamaTokenizer

DATASET_TASK_MAPPING = {
    'NLU': [""],
    'NER': ["COLL2023"]
}

    
class DatasetModule(Dataset):
    def __init__(self, data: list, system_message: str, roles: list, prefix: str, dataset_columns: tuple, 
                 tokenizer: LlamaTokenizer, data_args: dict) -> None:
        super(Dataset, self).__init__()
        self.data = data
        self.system_message = system_message
        self.roles = roles
        self.dataset_columns = dataset_columns
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        self.prefix = prefix
        self.task = data_args.task
        self.max_input_length = data_args.max_source_length
        self.max_output_length = data_args.max_target_length
        self.padding = "max_length" if data_args.pad_to_max_length else False,
        self.ignore_pad_token_for_loss = data_args.ignore_pad_token_for_loss

        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        example = self.data[index]
        example['task'] = self.task
        if self.task == 'pretrain':
            return self.PreprocessForPretrain(example)
        
        elif self.task == 'ner':
            return self.PreprocessForNER(example)
        
        elif self.task == 'nlu':
            return self.PreprocessForNLU(example)

        else:
            raise NotImplementedError
    

    def PreprocessForPreTrain(self, example):
        prompt, source, target = self.prompt_gather.ConstructPrompts(example)
        prompt = prompt.strip()
        
        prompt_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False) + self.tokenizer.encode(text=source, add_special_tokens=False)
        target_ids = self.tokenizer.encode(text=target+self.tokenizer.eos_token, add_special_tokens=False)
    
        if len(prompt_ids) > self.max_input_length:
            prompt_ids = prompt_ids[ :self.max_input_length]
        if len(target_ids) > self.max_output_length:
            target_ids = target_ids[ :self.max_output_length]
        
        input_ids_head = self.tokenizer(f'{self.system_message} USER:').input_ids
        input_ids_tail = self.tokenizer(f'ASSISTANT:', add_special_tokens=False).input_ids

        input_ids = input_ids_head + prompt_ids + input_ids_tail + target_ids
        input_attention_mask = [1] * len(input_ids_head + prompt_ids + input_ids_tail) + [1] * len(target_ids)
        
        context_length = len(input_ids_head + prompt_ids + input_ids_tail)    
        labels = [-100] * context_length + input_ids[context_length: ]
        
        if self.padding == 'max_length':
            pad_len = self.max_input_length + self.max_output_length - len(prompt_ids) - len(target_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            input_attention_mask = input_attention_mask + [0] * pad_len
            labels = labels + [self.tokenizer.pad_token_id] * pad_len
        
        if self.ignore_pad_token_for_loss:
            labels = [l if l != self.tokenizer.pad_token_id else -100 for l in labels]

        return {
            'input_ids': input_ids[:self.max_input_length],
            'attention_mask': input_attention_mask[:self.max_input_length],
            'labels': labels[:self.max_input_length]
        }
        
        # return {
        #     'input_ids': torch.tensor(input_ids).flatten(),
        #     'attention_mask': torch.tensor(input_attention_mask).flatten(),
        #     'labels': torch.tensor(labels).flatten()
        # }
    
    def PreprocessForNLU(self, example):
        inputs, targets = example[self.dataset_columns[0]], example[self.dataset_columns[1]]
        input_ids = self.tokenizer(text=inputs, add_special_tokens=False)['input_ids'][ :self.max_input_length-1]
        inv_input_ids = input_ids[::-1]
        
        attention_mask = [1]*(len(input_ids) + 1)
        pad_len = self.max_input_length - len(input_ids) - 1
        input_ids = [self.tokenizer.pad_token_id] * pad_len  + input_ids + [3]
        inv_input_ids = [self.tokenizer.pad_token_id] * pad_len + inv_input_ids + [3]
        attention_mask  = [0] * pad_len + attention_mask
        
        
        return {
        'input_ids': torch.tensor(input_ids).flatten(),
        'attention_mask': torch.tensor(attention_mask).flatten(),
        'inv_input_ids': torch.tensor(inv_input_ids).flatten(),
        'inv_attention_mask': torch.tensor(attention_mask).flatten(), 
        'labels': torch.tensor(targets).flatten(),
        'require_inv': torch.tensor(self.data_args.require_inv).flatten()}
    
    def PreprocessForNER(self, example):
        assert self.max_input_length == self.max_output_length, '''in task NER, the max length of sequence and ner tags should be same, but the input length: {0} with output: {1} tokens'''.format(self.max_input_length, self.max_output_length)
        
        tokens, ner_tags = example[self.dataset_columns[0]], example[self.dataset_columns[1]]
        token_ids = self.tokenizer(text=tokens, add_special_tokens=False)['input_ids'][ :self.max_input_length]
        ner_tags = ner_tags[:self.max_input_length]
        assert len(token_ids) == len(ner_tags), 'the number of tokenized token ids are different: {}!= {}'.format(len(token_ids), len(ner_tags))
        
        token_ids = [item[0] for item in token_ids]
        input_attention_mask = [1]*len(token_ids)
        
        padLen = self.max_input_length - len(token_ids)
        token_ids = [self.tokenizer.pad_token_id] * padLen + token_ids
        
        input_attention_mask = [0] * padLen + input_attention_mask
        ner_tags = [-100] * padLen + ner_tags
        
        assert len(ner_tags) == len(token_ids), 'the number of paddded tokenized token ids are different: {}!= {}'.format(len(token_ids), len(ner_tags))
        return {
                'input_ids': torch.tensor(token_ids).flatten(),
                'attention_mask': torch.tensor(input_attention_mask).flatten(),
                'inv_input_ids': torch.tensor(token_ids[::-1]).flatten(),
                'inv_attention_mask': torch.tensor(input_attention_mask[::-1]).flatten(), 
                'labels': torch.tensor(ner_tags).flatten(),
                'require_inv': torch.tensor(self.data_args.require_inv).flatten()}