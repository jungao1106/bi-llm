import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.nn import CrossEntropyLoss
from transformers import LlamaModel, LlamaForCausalLM
from peft import PeftModel

class LlamaForNLU(nn.Module):
    def __init__(self, model: [LlamaModel, nn.Module], args: Dict) -> None:
        super(LlamaForNLU, self).__init__()
        self.args = args
        self.config = model.config
        self.model = model
        self.cls_head = nn.Linear(model.config.hidden_size, args.num_labels)
        self.lossFN = CrossEntropyLoss()
        self.is_token_cls =(args.num_labels == 9)
    
    def forward(self, input_ids, attention_mask, inv_input_ids, inv_attention_mask, labels, require_inv = False):
        inv_last_hidden_states = None
        if isinstance(self.model, PeftModel):
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask = attention_mask)['last_hidden_state']
            
            if require_inv == True:
                inv_last_hidden_states = self.model(input_ids=inv_input_ids,
                                                    attention_mask=inv_attention_mask)['last_hidden_state']
        
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask = attention_mask)['last_hidden_state'].detach()

                if require_inv == True:
                    inv_last_hidden_states = self.model(input_ids=inv_input_ids,
                                                        attention_mask=inv_attention_mask)['last_hidden_state'].detach()

                    inv_last_hidden_states = inv_last_hidden_states.flip(dims=[1]) if self.is_token_cls else inv_last_hidden_states
        
        token_features = (last_hidden_states + inv_last_hidden_states) if inv_last_hidden_states is not None else last_hidden_states
        token_features = token_features if self.is_token_cls else token_features[:, -1, :]
        
        logits = self.cls_head(token_features.float())
        
        probs = F.softmax(logits, dim=-1)
        loss = self.lossFN(probs.view(-1, probs.shape[-1]), labels[:, 0].flatten())
        return (loss, probs)
        
      

class ModelModuleForCLS(nn.Module):
    def __init__(self, modelNameOrPath, mlppath, options, action) -> None:
        super(ModelModuleForCLS, self).__init__()
        self.__modelNameOrPath = modelNameOrPath
        self.__options = options
        if 'finetune' in action:
            self.__mlp = nn.Sequential(nn.Linear(4096, len(options)),
                                       nn.Softmax(dim=-1))
        else:
            self.__mlp = torch.load(mlppath, map_location='cpu')
            
        # if 'bi' in action:
        #     self.__model = AutoPeftModelForCausalLM.from_pretrained(modelNameOrPath).half()
        # else:
        self.__model = LlamaForCausalLM.from_pretrained(modelNameOrPath).half()
        for n, p in self.__model.named_parameters():
            p.requires_grad = False
        
        self.__lossFN = CrossEntropyLoss()
        
    @torch.no_grad()
    def inference(self, inputs):
        labels = inputs.pop('labels')[inputs['attention_mask'] != 0, ...].flatten()
        
        if 'inv_input_ids' in inputs.keys():
            invInputIds = inputs.pop('inv_input_ids')
            invAttnMask = inputs.pop('inv_attention_mask')
    
        with torch.no_grad():
            if len(self.__options) == 9: # ner
                if hasattr(self.__model, 'get_base_model'):
                    hidden_states = self.__model.get_base_model().model(**inputs)['last_hidden_state']
                    inv_hidden_states = self.__model.get_base_model().model(input_ids=invInputIds, 
                                                                            attention_mask=invAttnMask)['last_hidden_state']
                else:
                    hidden_states = self.__model.model(**inputs)['last_hidden_state']
                    inv_hidden_states = self.__model.model(input_ids=invInputIds, 
                                                                            attention_mask=invAttnMask)['last_hidden_state']
                inv_hidden_states = inv_hidden_states.flip(dims=[1])
                
                hidden_states = (hidden_states + inv_hidden_states)
            else: # sentiment cls
                hidden_states = self.__model.get_base_model().model(**inputs)['last_hidden_state'][:, -1, :]
            probs = self.__mlp(hidden_states.float())[inputs['attention_mask'] != 0, ...]
    
        return probs, labels
    
    def forward(self, inputs):
        labels = inputs.pop('labels').flatten()
        
        if 'inv_input_ids' in inputs.keys():
            invInputIds = inputs.pop('inv_input_ids')
            invAttnMask = inputs.pop('inv_attention_mask')
    
        with torch.no_grad():
            if len(self.__options) == 9: # ner
                if hasattr(self.__model, 'get_base_model'):
                    hidden_states = self.__model.get_base_model().model(**inputs)['last_hidden_state']
                    inv_hidden_states = self.__model.get_base_model().model(input_ids=invInputIds, 
                                                                            attention_mask=invAttnMask)['last_hidden_state']
                else:
                    hidden_states = self.__model.model(**inputs)['last_hidden_state']
                    inv_hidden_states = self.__model.model(input_ids=invInputIds,
                                                           attention_mask=invAttnMask)['last_hidden_state']
                inv_hidden_states = inv_hidden_states.flip(dims=[1])
                
                hidden_states = hidden_states + inv_hidden_states
            
            else: # sentiment cls
                hidden_states = self.__model.get_base_model().model(**inputs)['last_hidden_state'][:, -1, :]
        probs = self.__mlp(hidden_states.float()).view(-1, len(self.__options))
    
        return self.__lossFN(probs, labels)
    
    def SaveTrainedModel(self, val: float ,path: str = './') -> None:
        os.makedirs(os.path.join('/data/gj/Bi-LLaMA-new/vicuna-base-mlp', 'mlp-ner-norm-pos'), exist_ok=True)
        torch.save(self.__mlp, os.path.join('/data/gj/Bi-LLaMA-new/vicuna-base-mlp', 'mlp-ner-norm-pos/step{0}_mlp_loss_{1:.4f}.bin'.format(path, val)))