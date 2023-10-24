from transformers import Trainer
from typing import Union, Any, Dict, Optional, List, Tuple
import torch.nn as nn
import torch
import os
from transformers.deepspeed import deepspeed_init
from transformers.utils import is_safetensors_available, is_sagemaker_mp_enabled
from peft import AutoPeftModelForCausalLM, PeftModel
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
  
if is_safetensors_available():
    import safetensors.torch

CONFIG_NAME = "config.json"

WEIGHTS_NAME = "pytorch_model.bin"
SAFE_WEIGHTS_NAME = "model.safetensors"
PEFT_NAME = "adapter_model.bin"
SAFE_PEFT_NAME = "adapter_model.safetensors"

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
    
class NLUTrainer(Trainer):
    
    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get('labels', None).detach()[:, 0]
        loss, probs = self.compute_loss(model, inputs, return_outputs= not prediction_loss_only)
        preds = torch.argmax(probs, dim=-1)
        return (loss, None, None) if prediction_loss_only else (loss, preds, labels)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs['require_inv'] = True if inputs['require_inv'].count_nonzero() else False
        loss, probs = model(**inputs)
        
        return (loss, probs) if return_outputs else loss
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.model.cls_head.state_dict()
        if self.args.save_safetensors:
            safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
        else:
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))  
        
        if isinstance(self.model.model, PeftModel):
            self.model.model.save_pretrained(
                    output_dir, safe_serialization=self.args.save_safetensors
                )
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    def _load_best_model(self):
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_peft_path = os.path.join(self.state.best_model_checkpoint, PEFT_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_safe_peft_path = os.path.join(self.state.best_model_checkpoint, SAFE_PEFT_NAME)
        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if os.path.exists(best_model_path) or os.path.exists(best_safe_model_path):
            if self.deepspeed:
                if self.model_wrapped is not None:
                    # this removes the pre-hooks from the previous engine
                    self.model_wrapped.destroy()
                    self.model_wrapped = None

                # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
                deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                    self,
                    num_training_steps=self.args.max_steps,
                    resume_from_checkpoint=self.state.best_model_checkpoint,
                )
                self.model = deepspeed_engine.module
                self.model_wrapped = deepspeed_engine
                self.deepspeed = deepspeed_engine
                self.optimizer = optimizer
                self.lr_scheduler = lr_scheduler
            else:
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(self.state.best_model_checkpoint, "user_content.pt")):
                        # If the 'user_content.pt' file exists, load with the new smp api.
                        # Checkpoint must have been saved with the new smp api.
                        smp.resume_from_checkpoint(
                            path=self.state.best_model_checkpoint,
                            tag=WEIGHTS_NAME,
                            partial=False,
                            load_optimizer=False,
                        )
                    else:
                        # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                        # Checkpoint must have been saved with the old smp api.
                        if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                            state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                        else:
                            state_dict = torch.load(best_model_path, map_location="cpu")

                        state_dict["_smp_is_partial"] = False
                        load_result = model.load_state_dict(state_dict, strict=True)
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        state_dict = torch.load(best_model_path, map_location="cpu")
                        
                    # If the model is on the GPU, it still works!
                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    load_result = model.cls_head.load_state_dict(state_dict, False)
                    
                    if isinstance(model.model, PeftModel):
                        # We load the model state dict on the CPU to avoid an OOM error.
                        if self.args.save_safetensors and os.path.isfile(best_safe_peft_path):
                            state_dict = safetensors.torch.load_file(best_safe_peft_path, device="cpu")
                        else:
                            state_dict = torch.load(best_peft_path, map_location="cpu")
                        
                        peft_state_dict = safetensors.torch.load_file(best_safe_peft_path, device="cpu")
                        peft_state_dict = {k.replace('weight', 'default.weight'): v for k, v in peft_state_dict.items()}
                        
                        model.model.load_state_dict(peft_state_dict, False)

                if not is_sagemaker_mp_enabled():
                    self._issue_warnings_after_load(load_result)
                    
                    
    def load_cls_head_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)

        if not any(
            [os.path.isfile(f) for f in [weights_file, safe_weights_file]]
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")


        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    state_dict = torch.load(weights_file, map_location="cpu")
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    state_dict = torch.load(weights_file, map_location="cpu")

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.cls_head.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)