import torch
from torch import nn
from transformers import BertConfig, BertModel, ViTModel
from torchcrf import CRF
import torch.nn.functional as F
from .PT_Bert import create_progressive_trust_bert

class TPM_MI(torch.nn.Module):
    def __init__(self, label_list, args):
        super().__init__()
        self.num_labels = len(label_list)
        # text encoder
        bert_config = BertConfig.from_pretrained(args.bert_model)
        self.bert = create_progressive_trust_bert(
            pretrained_model_name=args.bert_model,
            modulation_layers=range(0,12)

        )
        # image encoder
        self.vit = ViTModel.from_pretrained(args.vit_model)
        self.config = self.bert.config
        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads

        # project images as prompts
        self.img_prompt_encoder = torch.nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.config.hidden_size)
        )
        # learnable postional embeddings
        self.temporalEmbedding = torch.nn.Embedding(4, 768)
        self.transformer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.fc = torch.nn.Linear(768, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.img_exp_len_per_image = args.max_image_exp_length  # 64
        self.text_len = args.max_seq_length  # 64
        self.text_exp_len = args.max_exp_length  # 128

        self.total_img_exp_len = self.img_exp_len_per_image * 4  # 256
        self.text_start_idx = self.total_img_exp_len  # 256
        self.text_end_idx = self.text_start_idx + self.text_len  # 320
        self.text_exp_start_idx = self.text_end_idx  # 320

    def get_prompt(self, img_features=None):
        bsz, img_len, _ = img_features.size()
        # (bsz, 4, 768)-> (bsz, 4, n_layer*2*hidden_size)
        past_key_values_vit = self.img_prompt_encoder(img_features)
        past_key_values_vit = past_key_values_vit.view(
            bsz,
            img_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        # (bsz, seq_len + seq_len2, 12*2, n_head(12), 64)
        past_key_values = past_key_values_vit
        # (12*2, bsz, n_head, len, 64)
        # 12*[2,bsz,n_head,len,64]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None
                , img_features=None, labels=None, exp_input_ids=None, exp_attention_mask=None, exp_token_type_ids=None,
                img_exp_attention_mask=None, img_exp_token_type_ids=None, img_exp_input_ids=None, images_sim=None):
        img_weights = F.softmax(images_sim, dim=-1)

        bsz, img_len, channels, height, width = img_features.shape
        # (bsz, 4, 3, 224, 224) -> (bsz*4, 3, 224, 224)
        pixel_values = img_features.reshape(bsz * img_len, channels, height, width)
        # (bsz*4, 3, 224, 224) -> (bsz*4, 768)
        img_features = self.vit(pixel_values=pixel_values)[1]
        # (bsz*4, 768) -> (bsz, 4, 768)
        img_features = img_features.reshape(bsz, img_len, -1)
        img_weights_expanded = img_weights.unsqueeze(-1)  # [bsz, 4, 1]
        img_features = img_features * img_weights_expanded
        temp_embeddings = self.temporalEmbedding(torch.arange(4).to(img_features.device))
        # (4, 768) -> (1, 4, 768)
        temp_embeddings = temp_embeddings.unsqueeze(0)
        img_features = img_features + temp_embeddings
        # (bsz, 4, 768)
        tempral_features = self.transformer(img_features)
        # img_features (bsz, 4, 768) -> past_key_values 12*[2, bsz, n_head, len, 64]
        past_key_values = self.get_prompt(tempral_features)

        # 以上代码处理图像特征

        # 获取解释文本的长度
        exp_seq_len = exp_input_ids.size(1)  # ← 在这里计算！
        text_seq_len = input_ids.size(1)  # 原始文本长度
        img_exp_seq_len = img_exp_input_ids.size(1)  # 图像解释长度
        # ==================== 新增：创建各类token的索引 ====================
        device = input_ids.device
        # 图像解释token索引 (位置 0 到 img_exp_seq_len-1)
        img_exp_indices = torch.arange(img_exp_seq_len, device=device)
        # 原始文本token索引 (位置 img_exp_seq_len 到 img_exp_seq_len+text_seq_len-1)
        text_indices = torch.arange(img_exp_seq_len, img_exp_seq_len + text_seq_len, device=device)
        # 文本解释token索引 (位置 img_exp_seq_len+text_seq_len 到最后)
        text_exp_indices = torch.arange(
            img_exp_seq_len + text_seq_len,
            img_exp_seq_len + text_seq_len + exp_seq_len,
            device=device
        )
        # ==================================================================
        # 开始处理解释文本
        combined_input_ids = torch.cat([img_exp_input_ids, input_ids, exp_input_ids], dim=1)
        combined_attention_mask = torch.cat([img_exp_attention_mask, attention_mask, exp_attention_mask], dim=1)
        combined_token_type_ids = torch.cat([img_exp_token_type_ids, token_type_ids, exp_token_type_ids], dim=1)

        # 处理图像的提示设置
        prompt_guids_length = past_key_values[0][0].size(2)
        bsz = attention_mask.size(0)
        img_prompt_mask = torch.ones((bsz, prompt_guids_length)).to(attention_mask.device)
        final_attention_mask = torch.cat([
            img_prompt_mask,  # 图像prompt的掩码
            combined_attention_mask  # 解释+原始文本的掩码
        ], dim=1)

        output_bert_result = self.bert(input_ids=combined_input_ids,
                                       attention_mask=final_attention_mask,
                                       token_type_ids=combined_token_type_ids,
                                       past_key_values=past_key_values,
                                       text_indices=text_indices,
                                       img_exp_indices=img_exp_indices
                                       )
        sequence_output = output_bert_result['last_hidden_state']

        text_start_idx = img_exp_seq_len
        text_end_idx = text_start_idx + text_seq_len

        # 验证：确保起始位置不超过序列总长度
        total_seq_len = sequence_output.size(1)
        if text_start_idx >= total_seq_len:
            raise ValueError(f"文本起始位置{text_start_idx}超出序列长度{total_seq_len}")

        # 只取原始文本部分的隐藏状态
        text_sequence_output = sequence_output[:, text_start_idx:text_end_idx, :]
        actual_text_len = text_sequence_output.size(1)
        if actual_text_len != text_seq_len:
            print(f"警告：取出的文本长度({actual_text_len})不等于原始文本长度({text_seq_len})")
            # 可能由于截断或padding，进行截断或padding
            if actual_text_len > text_seq_len:
                text_sequence_output = text_sequence_output[:, :text_seq_len, :]
            else:
                # 填充到正确长度
                padding_len = text_seq_len - actual_text_len
                padding = torch.zeros((bsz, padding_len, text_sequence_output.size(2)),
                                      device=text_sequence_output.device)
                text_sequence_output = torch.cat([text_sequence_output, padding], dim=1)

        # ==================== 6. 分类和CRF解码 ====================
        emissions = self.fc(text_sequence_output)  # (bsz, text_seq_len, num_labels)
        logits = self.crf.decode(emissions, attention_mask.byte())

        # ==================== 7. 损失计算 ====================
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return (logits, loss)

