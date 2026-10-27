import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import Optional, List, Tuple, Union


class TextGuidedModulator(nn.Module):
    """
    文本引导的图像解释调制器（插件模块）
    """

    def __init__(self, hidden_size: int, reduction: int = 4):
        super().__init__()
        self.hidden_size = hidden_size

        # 文本聚合器
        self.text_aggregator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // reduction),
            nn.Tanh(),
            nn.Linear(hidden_size // reduction, hidden_size)
        )

        # 门控生成器
        self.gate_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # 温和初始化
        self._init_weights()

    def _init_weights(self):
        for module in self.text_aggregator:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.002)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.gate_generator:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.002)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_features: [batch_size, num_img_tokens, hidden_size]
            text_features: [batch_size, num_text_tokens, hidden_size]
        Returns:
            modulated_features: [batch_size, num_img_tokens, hidden_size]
        """
        # 聚合文本特征
        text_pooled = text_features.mean(dim=1)  # [batch_size, hidden_size]
        text_summary = self.text_aggregator(text_pooled)  # [batch_size, hidden_size]
        text_summary = text_summary.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 扩展到图像token维度
        text_summary = text_summary.expand(-1, image_features.size(1), -1)

        # 拼接并生成门控
        combined = torch.cat([image_features, text_summary], dim=-1)
        gate = self.gate_generator(combined)

        return image_features * gate


class ProgressiveTrustBertModel(BertPreTrainedModel):
    """
    渐进式信任BERT模型
    在每一层后插入调制器，让调制后的表示参与后续计算
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)

        # 使用标准的BERT模型（但我们需要访问其内部组件）
        self.bert = BertModel(config)

        # 添加调制器插件
        self.modulator = TextGuidedModulator(config.hidden_size)

        # 配置：在哪些层应用调制（默认所有层）
        self.modulation_layers = list(range(config.num_hidden_layers))

        # 是否启用调制
        self.use_modulation = True

        # 初始化后处理
        self.post_init()

    def set_modulation_layers(self, layers: List[int]):
        """设置应用调制的层"""
        self.modulation_layers = layers

    def enable_modulation(self, enabled: bool = True):
        """启用或禁用调制"""
        self.use_modulation = enabled

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # 自定义参数
            img_exp_indices: Optional[torch.Tensor] = None,
            text_indices: Optional[torch.Tensor] = None,
            **kwargs  # 接受任意额外参数
    ):
        # 设置输出选项
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # 处理输入
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定 input_ids 和 inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("必须指定 input_ids 或 inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 处理 attention_mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        # 扩展 attention_mask 为 [batch_size, 1, 1, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 处理 head_mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask[:, None, None, None]
            elif head_mask.dim() == 2:
                head_mask = head_mask[:, None, None, :]
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # ========== 1. 嵌入层 ==========
        if inputs_embeds is None:
            inputs_embeds = self.bert.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds

        # 用于存储所有层的隐藏状态
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_trust_scores = ()  # 收集调制门值

        # ========== 2. 逐层处理 ==========
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 当前BERT层的前向传播
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]  # 当前层的输出

            # ========== 关键修改：在每层后应用调制 ==========
            if (self.use_modulation and
                    i in self.modulation_layers and
                    img_exp_indices is not None and
                    text_indices is not None):
                # 提取图像特征和文本特征
                img_features = hidden_states[:, img_exp_indices, :]
                text_features = hidden_states[:, text_indices, :]

                # 应用调制
                modulated_img = self.modulator(img_features, text_features)

                # 更新隐藏状态（这将作为下一层的输入）
                new_hidden = hidden_states.clone()
                new_hidden[:, img_exp_indices, :] = modulated_img
                hidden_states = new_hidden

                # 收集调制门值（可选）
                gate_values = self.modulator.gate_generator(
                    torch.cat(
                        [img_features, text_features.mean(dim=1).unsqueeze(1).expand(-1, img_features.size(1), -1)],
                        dim=-1)
                )
                all_trust_scores = all_trust_scores + (gate_values.mean(dim=1).mean(dim=-1),)

            # 收集注意力（如果需要）
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # ========== 3. Pooler 层 ==========
        pooled_output = self.bert.pooler(hidden_states)

        # ========== 4. 返回结果 ==========
        if not return_dict:
            return (hidden_states, pooled_output) + (all_hidden_states, all_attentions)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# ========== 工厂函数 ==========
def create_progressive_trust_bert(
        pretrained_model_name: str = 'bert-base-uncased',
        modulation_layers: Optional[List[int]] = None
) -> ProgressiveTrustBertModel:
    """
    创建渐进式信任BERT模型

    Args:
        pretrained_model_name: 预训练模型名称或路径
        modulation_layers: 应用调制的层索引列表，None表示所有层

    Returns:
        model: 加载了预训练权重的渐进式信任BERT模型
    """
    # 加载配置
    config = BertConfig.from_pretrained(pretrained_model_name)

    # 创建模型并加载预训练权重
    model = ProgressiveTrustBertModel.from_pretrained(
        pretrained_model_name,
        config=config
    )

    # 设置调制层
    if modulation_layers is not None:
        model.set_modulation_layers(modulation_layers)

    return model