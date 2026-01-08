#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Lightweight Memory-Driven Self-Attention for Hyperspectral Image Classification with CNN-Transformer
Cross-Feature Fusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1.0 / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        keys_transposed = keys.transpose(1, 2)
        energy = torch.bmm(query, keys_transposed)
        attention_weights = F.softmax(energy * self.scale, dim=2)
        linear_combination = torch.bmm(attention_weights, values)
        return linear_combination


class LightweightLSTMCell(nn.Module):
    
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(LightweightLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.num_features = 4
        self.padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=self.num_features * self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, hidden_state, cell_state):
        combined = torch.cat([x, hidden_state], dim=1)
        combined_conv = self.conv(combined)

        gate_i, gate_f, gate_o, gate_g = torch.split(
            combined_conv, 
            combined_conv.size(1) // self.num_features, 
            dim=1
        )

        input_gate = torch.sigmoid(gate_i)
        forget_gate = torch.sigmoid(gate_f)
        output_gate = torch.sigmoid(gate_o)
        candidate_state = torch.tanh(gate_g)
        
        new_cell_state = cell_state * forget_gate + input_gate * candidate_state
        new_hidden_state = output_gate * torch.tanh(new_cell_state)

        return new_hidden_state, new_cell_state, input_gate, forget_gate, output_gate, candidate_state

    @staticmethod
    def init_hidden(batch_size, hidden_channels, spatial_shape, device=None):
        height, width = spatial_shape
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        hidden_state = torch.zeros(batch_size, hidden_channels, height, width, device=device)
        cell_state = torch.zeros(batch_size, hidden_channels, height, width, device=device)
        
        return Variable(hidden_state), Variable(cell_state)


class SJMA(nn.Module):
    
    def __init__(self, input_channels, hidden_channels, kernel_size, bias, attention_size, device=None):
        super(SJMA, self).__init__()
        self.attention = Attention(attention_size)
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.bias = bias
        self.device = device

        self.conv3d = nn.Conv3d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(1, 1, 3),
            padding=(0, 0, 1),
            stride=(1, 1, 1),
            bias=False
        )
        
        self.conv2d = nn.Conv2d(
            in_channels=input_channels * 2, 
            out_channels=input_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )

        self.lstm_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            cell = LightweightLSTMCell(
                input_channels=self.input_channels[layer],
                hidden_channels=self.hidden_channels[layer],
                kernel_size=self.kernel_size,
                bias=self.bias
            )
            self.lstm_cells.append(cell)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        internal_states = []
        outputs = []
        
        for step in range(time_steps):
            current_input = x[:, step, :, :, :]
            channel_output = self.conv3d(current_input.unsqueeze(1)).squeeze(1)
            
            for layer in range(self.num_layers):
                if step == 0:
                    h, c = LightweightLSTMCell.init_hidden(
                        batch_size, 
                        self.hidden_channels[layer], 
                        (height, width),
                        device=self.device
                    )
                    internal_states.append((h, c))
                
                h, c = internal_states[layer]
                current_input, c, i, f, o, g = self.lstm_cells[layer](current_input, h, c)
                internal_states[layer] = (current_input, c)
            
            outputs.append(current_input)
        
        final_output = outputs[-1]
        
        query, keys, values = self._generate_qkv(final_output, c, i, f, g, o)
        
        query = query.view(batch_size, -1, height * width)
        keys = keys.view(batch_size, -1, height * width)
        values = values.view(batch_size, -1, height * width)
        
        attention_output = self.attention(query, keys, values)
        attention_output = attention_output.view(batch_size, -1, height, width)
        
        output = self.conv2d(torch.cat([attention_output, channel_output], dim=1))
        return output
    
    def _generate_qkv(self, h_states, c_states, i_states, f_states, g_states, o_states):
        values = h_states
        query = (c_states + f_states) / 2
        keys = (g_states + h_states) / 2
        return query, keys, values


class SpatialAttention(nn.Module):
    
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.attention_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention_block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.attention_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention_block(x)
    

class S2CMA(nn.Module):
    
    def __init__(self, channels=512, attention_bias=False, dropout_rate=0.0):
        super(S2CMA, self).__init__()
        self.qkv_conv = nn.Conv2d(channels, 3 * channels, 1, stride=1, padding=0, bias=attention_bias)
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention(channels)
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        q, k, v = self.qkv_conv(x).chunk(3, dim=1)
        
        q_attended = self.channel_attention(q)
        k_attended = self.spatial_attention(k)
        
        fused_features = self.fusion_conv(torch.cat([q_attended, k_attended], dim=1))
        output = fused_features * v
        output = self.dropout(output)
        
        return output


class MDSANetBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dims, kernel_size, patch_size, device=None):
        super(MDSANetBlock, self).__init__()

        self.sjma = SJMA(
            input_channels=input_dim,
            hidden_channels=hidden_dims,
            kernel_size=kernel_size,
            bias=True,
            attention_size=patch_size * patch_size,
            device=device
        )
        
        self.s2cma = S2CMA(channels=input_dim)

        self.aap_sjma = nn.AdaptiveAvgPool2d(1)
        self.aap_s2cma = nn.AdaptiveAvgPool2d(1)

        self.fc_sjma = nn.Sequential(
            nn.Linear(input_dim, input_dim // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 16, input_dim, bias=False),
            nn.Sigmoid()
        )
        
        self.fc_s2cma = nn.Sequential(
            nn.Linear(input_dim, input_dim // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 16, input_dim, bias=False),
            nn.Sigmoid()
        )
        
        self.conv3x3 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)
        self.conv5x5 = nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=1, padding=2, groups=input_dim)

        self.output_conv = nn.Conv2d(2 * input_dim, output_dim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):

        batch_size, channels, height, width = x.size()
        global_features = self.sjma(x.unsqueeze(1))    
        local_features = self.s2cma(x)

        sjma_attention = self.fc_sjma(
            self.aap_sjma(global_features).view(batch_size, channels))
        sjma_attention = F.softmax(sjma_attention, dim=1)
        conv_attention = self.fc_s2cma(
            self.aap_s2cma(local_features).view(batch_size, channels))
        conv_attention = F.softmax(conv_attention, dim=1)
        fusion_attention = torch.matmul(sjma_attention.T, conv_attention)

        sjma_enhanced = torch.matmul(
            fusion_attention.T, 
            global_features.view(batch_size, channels, -1)
        ).view(batch_size, channels, height, width)
        
        s2cma_enhanced = torch.matmul(
            fusion_attention, 
            local_features.view(batch_size, channels, -1)
        ).view(batch_size, channels, height, width)
        
        global_features = global_features + sjma_enhanced
        local_features = local_features + s2cma_enhanced
        
        x3 = self.conv3x3(global_features)
        x5 = self.conv5x5(local_features)

        temporal_final = global_features + x3
        conv_final = local_features + x5
        
        output = self.output_conv(torch.cat([temporal_final, conv_final], dim=1))
        return output


class MDSANet(nn.Module):
    def __init__(self, input_channels, num_classes, patch_size, device=None, use_dropout=True):
        super(MDSANet, self).__init__()
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.feature_extractor = MDSANetBlock(
            input_dim=input_channels,
            output_dim=32,
            hidden_dims=[input_channels],
            kernel_size=1,
            patch_size=patch_size,
            device=self.device
        )
        
        self.batch_norm = nn.BatchNorm2d(32)
        self.activation = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes, bias=False)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        features = self.feature_extractor(x)
        features = self.batch_norm(features)
        features = self.activation(features)

        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(pooled_features)
        
        return output


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    test_input = torch.randn(64, 11, 11, 32).to(device)
    
    model = MDSANet(input_channels=32, num_classes=15, patch_size=11, device=device).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    print(f'Trainable parameters: {trainable_params / 1e6:.2f}M')
    
    try:
        from calflops import calculate_flops
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=(64, 11, 11, 32),
            output_as_string=True,
            output_precision=4
        )
        print(f"Model FLOPs: {flops}, MACs: {macs}, Parameters: {params}")
    except ImportError:
        print("calflops library not installed, skipping FLOPs calculation")
    
    with torch.no_grad():
        output = model(test_input)
        print(f'Output shape: {output.shape}')


if __name__ == '__main__':
    main() 

