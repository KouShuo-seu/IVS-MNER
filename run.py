import os
from xmlrpc.client import Boolean

import torch
import argparse
import logging
import numpy as np
import random
from torch.utils.data import DataLoader
from modules.dataset import TextSperateImageVITCRFDataset
from modules.trainer import NERTrainer
from models.BERTModels_Gated import TPM_MI

logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=1234):
    """set random seed"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        default='./dataset/text')
    parser.add_argument('--image_path',
                        type=str,
                        default='./dataset/images')
    parser.add_argument('--twitter2017_image_path',
                        type=str,
                        default='./dataset/twitter2017_images')
    parser.add_argument('--bert_model',
                        type=str,
                        default='./pretrained_models/bert-base-uncased')
    parser.add_argument('--vit_model',
                        type=str,
                        default='./pretrained_models/ViTB-16')
    parser.add_argument('--num_epochs',
                        default=15,
                        type=int,
                        help="num training epochs"),

    
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help="cuda or cpu")
    parser.add_argument('--batch_size',
                        default=8,
                        type=int,
                        help="batch size")
    parser.add_argument('--lr',
                        default=5e-5,
                        type=float,
                        help="learning rate")
    parser.add_argument('--bert_lr_scale',
                        default=1,
                        type=float,
                        help="BERT部分的学习率缩放因子（相对于base lr）")
    parser.add_argument('--new_module_lr_scale',
                        default=5.0,
                        type=float,
                        help="新增模块（调制器）的学习率缩放因子")
    parser.add_argument('--warmup_ratio',
                        default=0.01,
                        type=float)
    parser.add_argument('--seed',
                        default=1234,
                        type=int,
                        help="random seed, default is 1")
    parser.add_argument('--max_seq_length',
                        default=64,
                        type=int)
    # MI is MNER-MI and UNI (uniform) is MNER-MI-Plus
    parser.add_argument('--dataset',
                        default="MI-withIS",
                        choices=["MI-withIS", "MI_plus-withIS"],
                        type=str)
    # 新增：explaination的最大长度
    parser.add_argument('--max_exp_length',
                        type=int,
                        default=128,
                        help="explaination最大序列长度")
    parser.add_argument('--max_image_exp_length',
                        type=int,
                        default=64,
                        help="image_explaination最大序列长度")
    args = parser.parse_args()

    # log file path
    os.makedirs('./logs', exist_ok=True)
    fileHandler = logging.FileHandler(f'./logs/log.txt', mode='a', encoding='utf8')
    file_format = logging.Formatter('%(asctime)s - %(levelname)s -   %(message)s')
    fileHandler.setFormatter(file_format)
    logger = logging.getLogger(__name__)
    logger.addHandler(fileHandler)

    for k, v in vars(args).items():
        logger.info(" " + str(k) + " = %s", str(v))

    set_seed(args.seed)

    # 创建数据集
    train_dataset = TextSperateImageVITCRFDataset(args, f'MNER-{args.dataset}_train.txt')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_dataset = TextSperateImageVITCRFDataset(args, f'MNER-{args.dataset}_val.txt')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dataset = TextSperateImageVITCRFDataset(args, f'MNER-{args.dataset}_test.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    label_mapping = train_dataset.processor.get_label_crf_mapping()
    label_list = list(label_mapping.keys())

    # 传递args给模型，使其知道是否使用explaination
    model = TPM_MI(label_list, args)

    # ===== 新增：使用 DataParallel 包装模型 =====
    if torch.cuda.device_count() > 1 and args.device == 'cuda':
        model = torch.nn.DataParallel(model)
    # =========================================

    # 检查数据集是否返回正确数量的元素
    sample_item = train_dataset[0]
    logger.info(f"数据集返回的元素数量: {len(sample_item)}")
    logger.info(f"使用explanation功能，最大explanation长度: {args.max_exp_length}")
    logger.info(f"使用image_explanation功能，最大image_explanation长度: {args.max_image_exp_length}")
    logger.info(f"解释文本input_ids形状: {sample_item[5].shape}")

    trainer = NERTrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader,
                         model=model, label_map=label_mapping, args=args, logger=logger)
    trainer.train()
