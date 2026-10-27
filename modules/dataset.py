import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTImageProcessor
from io import BytesIO
import argparse


class MNERProcessor:
    def __init__(self, args) -> None:
        self.data_path = args.data_path
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    #读入了explaination，下一步处理是进行分词
    def load_from_file(self, file_name):
        words, labels, images, explanations, images_expls, images_sims = [], [], [], [], [],[]
        with open(os.path.join(self.data_path, file_name), 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                line_json = json.loads(line)
                word = line_json["text"]
                label = line_json["label"]
                explanation = line_json["explaination"]
                images_expl = line_json["images_expl"]
                images_sim = line_json["images_sim"]
                for j, l in enumerate(label):
                    # entity label is MISC in twitter2017, but MIS in MNER-MI
                    if l == "B-MISC":
                        label[j] = "B-MIS"
                    elif l == "I-MISC":
                        label[j] = "I-MIS"
                image = line_json["images"]

                words.append(word)
                labels.append(label)
                images.append(image)
                explanations.append(explanation)
                images_expls.append(images_expl)
                images_sims.append(images_sim)
            assert len(words) == len(labels) == len(images) == len(explanations)== len(images_expls)==len(images_sims)

        return {
            "words": words,
            "labels": labels,
            "images": images,
            "explanations": explanations,
            "images_expls": images_expls, # 修改5: 返回images_expls
            "images_sims": images_sims
        }

    # use softmax as decoder
    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MIS", "I-MIS", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 0)}
        return label_mapping

    # use crf as decoder
    def get_label_crf_mapping(self):
        LABEL_LIST = ["O", "B-MIS", "I-MIS", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 0)}
        return label_mapping


# Process each image separately, and use crf as decoder, and VIT as image encoder
class TextSperateImageVITCRFDataset(Dataset):
    def __init__(self, args, file_name) -> None:
        self.max_seq_length = args.max_seq_length
        self.max_exp_length = args.max_exp_length
        self.max_img_exp_length =args.max_image_exp_length
        self.processor = MNERProcessor(args)
        # VIT processor for VIT encoder
        self.vit_processor = ViTImageProcessor.from_pretrained(args.vit_model)
        self.data_dict = self.processor.load_from_file(file_name)
        self.tokenizer = self.processor.tokenizer
        self.label_mapping = self.processor.get_label_crf_mapping()
        # MNER-MI image path
        self.image_path = args.image_path
        # twitter-2017 image path (if use MNER-MI-Plus dataset)
        self.twitter2017_image_path = args.twitter2017_image_path


    def __len__(self):
        return len(self.data_dict["words"])

    def __getitem__(self, idx):
        word_list = self.data_dict["words"][idx]
        label_list = self.data_dict["labels"][idx]
        images = self.data_dict["images"][idx]
        explanation_list = self.data_dict["explanations"][idx]
        images_expl_list = self.data_dict["images_expls"][idx]# 新增
        images_sim_list = self.data_dict["images_sims"][idx]
        if not explanation_list or len(explanation_list) == 0:
            explanation_list = []  # 或 ["no explanation"]
        if not images_expl_list or len(images_expl_list) == 0:
            images_expl_list = [[] for _ in range(len(images))]
        images_sim_tensor = torch.tensor(images_sim_list, dtype=torch.float32)
        # (4, 3, 224, 224); initialize the image representation with all zero tensors
        seperate_images_feature = torch.zeros((4, 3, 224, 224))
        for i, image in enumerate(images):
            if "twitter2017" in image:
                img_path = os.path.join(self.twitter2017_image_path, image.split('-')[1])
            else:
                img_path = os.path.join(self.image_path, image)
            try:
                image = Image.open(img_path).convert("RGB")
                pixels = self.vit_processor(images=image, return_tensors='pt')["pixel_values"]
                seperate_images_feature[i] = pixels
            except OSError:
                try:
                    with open(img_path, 'rb') as f:
                        # some images is corrupted and missing JPEG end markers, so manually add \xff and \xd9
                        f = f.read()
                        f = f + B'\xff' + B'\xd9'
                    image = Image.open(BytesIO(f)).convert("RGB")
                    pixels = self.vit_processor(images=image, return_tensors='pt')["pixel_values"]
                    seperate_images_feature[i] = pixels
                except:
                    # if the image does not exist, the altername image is used
                    img_path = os.path.join(self.image_path, 'inf.png')
                    image = Image.open(img_path).convert("RGB")
                    pixels = self.vit_processor(images=image, return_tensors='pt')["pixel_values"]
                    seperate_images_feature[i] = pixels

        tokens, labels = [], []
        for i, word in enumerate(word_list):
            # a word may contain multiple tokens
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    # the extra tokends labels are X
                    labels.append(self.label_mapping["X"])

        if len(tokens) >= self.max_seq_length - 1:
            tokens = tokens[0:(self.max_seq_length - 2)]
            labels = labels[0:(self.max_seq_length - 2)]

        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq_length, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
            encode_dict['attention_mask']
        labels = [self.label_mapping["X"]] + labels + [self.label_mapping["X"]] + [self.label_mapping["O"]] * (
                    self.max_seq_length - len(labels) - 2)

        # 新增：处理explaination、images_exp
        if len(images_expl_list) < 4:
            images_expl_list = images_expl_list + [["UNK"] for _ in range(4 - len(images_expl_list))]#"no", "explanation"
        else:
            images_expl_list = images_expl_list[:4]  # 只取前4个
        # 直接构建三个张量列表
        all_img_exp_input_ids = []
        all_img_exp_token_type_ids = []
        all_img_exp_attention_mask = []

        # 对每个image_expl单独处理
        for img_expl in images_expl_list:
            img_exp_tokens = []
            if not img_expl or len(img_expl) == 0:
                img_expl = ["[UNK]"]  # 分词后的默认文本
            for word in img_expl:
                token = self.tokenizer.tokenize(word)
                img_exp_tokens.extend(token)
            if len(img_exp_tokens) >= self.max_img_exp_length - 1:
                img_exp_tokens = img_exp_tokens[0:(self.max_img_exp_length - 2)]

            img_exp_encode_dict = self.tokenizer.encode_plus(
                    img_exp_tokens,
                    max_length=self.max_img_exp_length,
                    truncation=True,
                    padding='max_length'
                )

            all_img_exp_input_ids.extend(img_exp_encode_dict['input_ids'])
            all_img_exp_token_type_ids.extend(img_exp_encode_dict['token_type_ids'])
            all_img_exp_attention_mask.extend(img_exp_encode_dict['attention_mask'])

        # 3. 堆叠为张量
        img_exp_input_ids = torch.tensor(all_img_exp_input_ids)  # 形状: (4*max_img_exp_length,)
        img_exp_token_type_ids = torch.tensor(all_img_exp_token_type_ids)
        img_exp_attention_mask = torch.tensor(all_img_exp_attention_mask)

        exp_tokens = []
        if not explanation_list or len(explanation_list) == 0:
            explanation_list = ["UNK"]#"no", "explanation"]
        for i, exp in enumerate(explanation_list):
            token = self.tokenizer.tokenize(exp)
            exp_tokens.extend(token)
        if len(exp_tokens) >= self.max_exp_length - 1:
            exp_tokens = exp_tokens[0:(self.max_exp_length - 2)]

        exp_encode_dict = self.tokenizer.encode_plus(
                exp_tokens,
                max_length=self.max_exp_length,
                truncation=True,
                padding='max_length'
            )
        exp_input_ids, exp_token_type_ids, exp_attention_mask = exp_encode_dict['input_ids'], exp_encode_dict['token_type_ids'], \
            exp_encode_dict['attention_mask']
        # 返回所有数据，包括新增的explaination编码

        return torch.tensor(input_ids),torch.tensor(token_type_ids),torch.tensor(attention_mask),seperate_images_feature,\
torch.tensor(labels), torch.tensor(exp_input_ids), \
            torch.tensor(exp_token_type_ids),  \
            torch.tensor(exp_attention_mask),torch.tensor(idx), \
            img_exp_input_ids,img_exp_token_type_ids,img_exp_attention_mask,images_sim_tensor


