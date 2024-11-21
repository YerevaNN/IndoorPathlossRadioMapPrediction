from typing import Literal, Optional

import torch
import torch.nn as nn

from networks.vit_pp import ViTPlusPlus
from networks.step_neck import StepNeck
from networks.upernet import FPN_fuse, PSPModule
from utils import unpatch


class ViTPlusPlusUPerNet(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        image_size: int,
        mlp_input_dim: int,
        min_mlp_tokens: int,
        num_channels: int,
        v_num_channels: int,
        v_patch_size: int,
        v_hidden_size: int,
        v_num_hidden_layers: int,
        v_num_attention_heads: int,
        mixer_out: int,
        res_hidden_states: list[int],
        use_upernet: bool,
        up_pool_scales: list[int],
        neck_input_dim: int,
        neck_scales: list[float],
        neck_size: list[int],
        pre_out_channels: int,
        model_type: Literal["clip", "dino_v2"],
        pretrained: str,
    ):
        super().__init__()
        
        assert (
            mixer_out is None or int(mixer_out ** 0.5) ** 2 == mixer_out,
            "Mixer's output size should be a square of a whole number"
        )
        
        self.v_hidden_size = v_hidden_size
        self.v_patch_size = v_patch_size
        
        self.conv = nn.Conv2d(num_channels, v_num_channels, kernel_size=3, padding="same")
        
        self.vit_pp = ViTPlusPlus(
            mlp_input_dim=mlp_input_dim, image_size=image_size,
            v_num_channels=v_num_channels, v_patch_size=v_patch_size,
            v_hidden_size=v_hidden_size, v_num_hidden_layers=v_num_hidden_layers,
            v_num_attention_heads=v_num_attention_heads, model_type=model_type, pretrained=pretrained,
        )
        self.res_hidden_states = res_hidden_states
        self.use_upernet = use_upernet
        
        feature_channels = [v_hidden_size] * (v_num_hidden_layers + 2)
        if res_hidden_states is not None:
            feature_channels = [feature_channels[i] for i in res_hidden_states]
        
        self.num_tokes = (image_size // v_patch_size) ** 2
        self.mixer_out = mixer_out or self.num_tokes
        self.mixers = nn.ModuleList(
            [
                nn.Conv1d(self.num_tokes + min_mlp_tokens + 1, self.mixer_out, 1) for _ in range(len(feature_channels))
            ]
        )
        if not self.use_upernet:
            self.bns = nn.ModuleList([nn.BatchNorm2d(c) for c in feature_channels])
        
        # Neck
        self.neck_input_dim = neck_input_dim
        if neck_scales:
            self.pre_necks = nn.ModuleList(
                [
                    nn.Linear(v_hidden_size, neck_input_dim) for _ in range(len(feature_channels))
                ]
            )
            self.neck = StepNeck(
                in_channels=[neck_input_dim] * len(feature_channels),
                out_channels=neck_size,
                scales=neck_scales
            )
            feature_channels = neck_size
        else:
            self.neck = None
        
        # UperNet
        if self.use_upernet:
            self.PPN = PSPModule(feature_channels[-1] + 2 * self.contest, bin_sizes=up_pool_scales)
            self.FPN = FPN_fuse([fc + 2 * self.contest for fc in feature_channels])
        
        if feature_channels[0] // (v_patch_size ** 2) == feature_channels[0] / (v_patch_size ** 2):
            head_out = feature_channels[0] // (v_patch_size ** 2)
        elif pre_out_channels:
            head_out = pre_out_channels
        else:
            head_out = feature_channels[0] // self.num_tokes
        
        self.unpatch_match = pre_out_channels and nn.Conv2d(
            v_hidden_size, pre_out_channels * v_patch_size ** 2, kernel_size=1,
            padding="same"
        )
        self.head = nn.Conv2d(
            feature_channels[0] + 2 * self.contest if self.neck else head_out,
            num_classes, kernel_size=3, padding="same"
        )
    
    def forward(
        self, image, sequence=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        vit_out = self.vit_pp(
            self.conv(image), sequence,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        vit_hidden_states = vit_out.hidden_states
        
        h = w = int(self.mixer_out ** 0.5)
        
        j = 0
        output = []
        for i, v in enumerate(vit_hidden_states + (vit_out.last_hidden_state,)):
            if self.res_hidden_states and i not in self.res_hidden_states:
                continue
            m = self.mixers[j]
            bn = self.bns[j] if not self.use_upernet else lambda x: x
            pn = self.pre_necks[j] if self.neck else lambda x: x
            j += 1
            depth = self.neck_input_dim or self.v_hidden_size
            o = m(v[:, :m.in_channels])
            o = pn(o).reshape(-1, h, w, depth).permute(0, 3, 1, 2)
            o = bn(o)
            output.append(o)
        
        if self.use_upernet:
            if self.neck:
                output = list(self.neck(output))
            # Up path
            output = [
                torch.cat([o, torch.nn.functional.interpolate(image[:, :2], o.shape[2:])], dim=1)
                for o in output
            ]
            output[-1] = self.PPN(output[-1])
            output = self.FPN(output)
        else:
            # output = torch.stack(output).mean(dim=0)
            output = torch.stack(output).mean(dim=0)
        
        # matching the embedding dimension for proper unpatching
        if self.unpatch_match:
            output = self.unpatch_match(output)
        
        if not self.neck:
            h = w = int(self.num_tokes ** 0.5)
            # unpatching the output
            output = unpatch(output, h, w, self.head.in_channels, self.v_patch_size)
        
        return self.head(output)
