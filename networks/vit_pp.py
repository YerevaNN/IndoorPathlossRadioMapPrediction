from typing import Optional

import torch
import torch.nn as nn
from transformers import Dinov2Config, Dinov2Model
from transformers.modeling_outputs import BaseModelOutputWithPooling


class ViTPlusPlus(nn.Module):
    
    def __init__(
        self, mlp_input_dim: int, image_size: int, v_num_channels: int, v_patch_size: int,
        v_hidden_size: int, v_num_hidden_layers: int, v_num_attention_heads: int, pretrained: str,
    ):
        super().__init__()
        self.v_num_layers = v_num_hidden_layers
        vit_config = dict(
            image_size=image_size,
            num_channels=v_num_channels, patch_size=v_patch_size,
            hidden_size=v_hidden_size, num_hidden_layers=v_num_hidden_layers,
            num_attention_heads=v_num_attention_heads, output_hidden_states=True
        )
        self.vit = Dinov2Model(Dinov2Config(**vit_config))
        if pretrained:
            self.vit: Dinov2Model = self.vit.from_pretrained(pretrained, output_hidden_states=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, v_hidden_size),
        )
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        sequence: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions or self.vit.config.output_attentions
        output_hidden_states = output_hidden_states or self.vit.config.output_hidden_states
        return_dict = return_dict or self.vit.config.use_return_dict
        
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        img_embeddings = self.vit.embeddings(pixel_values, bool_masked_pos=None)
        
        mask = None
        embeddings = img_embeddings
        
        encoder_params = dict(
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        encoder_params["hidden_states"] = embeddings
        encoder_params["head_mask"] = None if mask is None else (
            mask.unsqueeze(0).repeat(self.v_num_layers, 1, 1, 1).unsqueeze(2)
        )
        encoder_outputs = self.vit.encoder(**encoder_params)
        
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
