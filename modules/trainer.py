import json

import torch
from torch import optim
from tqdm import tqdm
from seqeval.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
from ner_evaluate import evaluate
from torch.utils import data


class NERTrainer:
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, label_map=None,
                 args=None, logger=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.logger = logger
        self.label_map = label_map
        self.refresh_step = 2
        self.best_train_metric = 0
        self.best_train_epoch = 0
        self.best_dev_metric = 0
        self.best_dev_epoch = 0
        self.optimizer = None

        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args

    def train(self):
        self.bert_before_train()

        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Using explanation")
        self.logger.info("  Max explanation length = {}".format(self.args.max_exp_length))
        self.logger.info("  Max image_explanation length = {}".format(self.args.max_image_exp_length))
        pbar = tqdm(total=self.train_num_steps)
        avg_loss = 0

        for epoch in range(1, self.args.num_epochs + 1):
            y_true, y_pred = [], []
            y_true_idx, y_pred_idx = [], []

            pbar.set_description(f"Epoch {epoch}/{self.args.num_epochs + 1}")
            self.model.train()

            for batch in self.train_data:
                self.step += 1

                batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                attention_mask, labels, logits, loss,idds = self._step(batch)
                avg_loss += loss.detach().cpu().item()

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if isinstance(logits, torch.Tensor):  # CRF return lists
                    logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                label_ids = labels.to('cpu').numpy()
                input_mask = attention_mask.to('cpu').numpy()
                label_map = {idx: label for label, idx in self.label_map.items()}
                reverse_label_map = {label: idx for label, idx in self.label_map.items()}

                for row, mask_line in enumerate(input_mask):
                    true_label = []
                    true_label_idx = []
                    true_predict = []
                    true_predict_idx = []
                    for column, mask in enumerate(mask_line):
                        if column == 0:
                            continue
                        if mask:
                            if label_map[label_ids[row][column]] != "X":
                                true_label.append(label_map[label_ids[row][column]])
                                true_label_idx.append(label_ids[row][column])
                                true_predict.append(label_map[logits[row][column]])
                                true_predict_idx.append(logits[row][column])
                        else:
                            break
                    y_true.append(true_label)
                    y_true_idx.append(true_label_idx)
                    y_pred.append(true_predict)
                    y_pred_idx.append(true_predict_idx)

                if self.step % self.refresh_step == 0:
                    avg_loss = float(avg_loss) / self.refresh_step
                    pbar.set_postfix(loss=avg_loss)
                    avg_loss = 0
                    pbar.update(self.refresh_step)

            results = classification_report(y_true, y_pred, digits=4)


            self.logger.info("***** Train Eval results *****")
            self.logger.info("\n%s", results)
            f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])

            if f1_score > self.best_train_metric:
                self.best_train_metric = f1_score
                self.best_train_epoch = epoch

            self.logger.info(f"Epoch {epoch}/{self.args.num_epochs}, best train f1: {self.best_train_metric},\
                            best train epoch: {self.best_train_epoch}, current train f1 score: {f1_score}")

            self.evaluate(epoch)  # generator to dev.

        pbar.close()
        self.pbar = None
        self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                                self.best_dev_metric))

        self.test()

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        y_true_idx, y_pred_idx = [], []
        step = 0
        errorArr =[]
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            for batch in self.dev_data:
                step += 1
                batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                attention_mask, labels, logits, loss,idds = self._step(batch)
                total_loss += loss.detach().cpu().item()

                if isinstance(logits, torch.Tensor):
                    logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                label_ids = labels.detach().cpu().numpy()
                senIds = idds.to('cpu').numpy()
                input_mask = attention_mask.detach().cpu().numpy()
                label_map = {idx: label for label, idx in self.label_map.items()}
                for row, mask_line in enumerate(input_mask):
                    sample_idx = senIds[row]
                    word_list = self.dev_data.dataset.data_dict["words"][sample_idx]
                    current_sentence =" ".join(word_list)
                    true_label = []
                    true_label_idx = []
                    true_predict = []
                    true_predict_idx = []
                    need_rec=False
                    for column, mask in enumerate(mask_line):
                        if column == 0:
                            continue
                        if mask:
                            if label_map[label_ids[row][column]] != "X":
                                true_label.append(label_map[label_ids[row][column]])
                                true_label_idx.append(label_ids[row][column])
                                true_predict.append(label_map[logits[row][column]])
                                true_predict_idx.append(logits[row][column])
                                pred_label = label_map[logits[row][column]]
                                true_label_val = label_map[label_ids[row][column]]
                                if pred_label != true_label_val:
                                    need_rec = True  # 标记需要记录
                        else:
                            break
                    if need_rec:
                        errorRec={
                            "sentence": current_sentence,
                            "true_label": true_label,
                            "true_predict":true_predict,

                        }
                        errorArr.append(errorRec)
                    y_true.append(true_label)
                    y_true_idx.append(true_label_idx)
                    y_pred.append(true_predict)
                    y_pred_idx.append(true_predict_idx)

            results = classification_report(y_true, y_pred, digits=4)
            self.logger.info("***** Dev Eval results *****")
            self.logger.info("\n%s", results)
            f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])

            if f1_score >= self.best_dev_metric:  # this epoch get best performance
                self.logger.info("Get better performance at epoch {}".format(epoch))
                self.best_dev_epoch = epoch
                self.best_dev_metric = f1_score  # update best metric(f1 score)
                torch.save(self.model.state_dict(), "./model.pt")

            self.logger.info(f"Epoch {epoch}/{self.args.num_epochs}, best dev f1: {self.best_dev_metric},\
                            best dev epoch: {self.best_dev_epoch}, current dev f1 score: {f1_score}")
            # if errorArr:
            #     self.logger.info(f"本轮评估共有 {len(errorArr)} 个识别错误:")
            #     for error in errorArr:
            #         for key, value in error.items():
            #             self.logger.info(f"{key} : {value}\n")

        self.model.train()

    def test(self):
        self.model.load_state_dict(torch.load("./model.pt"))
        self.model.to(self.args.device)
        self.model.eval()
        self.logger.info("***** Running test *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        y_true_idx, y_pred_idx = [], []
        step = 0
        errorArr = []
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            for batch in self.test_data:
                step += 1
                batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                attention_mask, labels, logits, loss,idds = self._step(batch)
                total_loss += loss.detach().cpu().item()

                if isinstance(logits, torch.Tensor):
                    logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                label_ids = labels.detach().cpu().numpy()
                senIds = idds.to('cpu').numpy()
                input_mask = attention_mask.detach().cpu().numpy()
                label_map = {idx: label for label, idx in self.label_map.items()}
                for row, mask_line in enumerate(input_mask):
                    sample_idx = senIds[row]
                    word_list = self.test_data.dataset.data_dict["words"][sample_idx]
                    current_sentence = " ".join(word_list)
                    true_label = []
                    true_label_idx = []
                    true_predict = []
                    true_predict_idx = []
                    need_rec = False
                    for column, mask in enumerate(mask_line):
                        if column == 0:
                            continue
                        if mask:
                            if label_map[label_ids[row][column]] != "X":
                                true_label.append(label_map[label_ids[row][column]])
                                true_label_idx.append(label_ids[row][column])
                                true_predict.append(label_map[logits[row][column]])
                                true_predict_idx.append(logits[row][column])
                                pred_label = label_map[logits[row][column]]
                                true_label_val = label_map[label_ids[row][column]]
                                if pred_label != true_label_val:
                                    need_rec = True  # 标记需要记录
                        else:
                            break
                    if need_rec:
                        errorRec={
                            "sentence": current_sentence,
                            "true_label": true_label,
                            "true_predict":true_predict,

                        }
                        errorArr.append(errorRec)
                    y_true.append(true_label)
                    y_true_idx.append(true_label_idx)
                    y_pred.append(true_predict)
                    y_pred_idx.append(true_predict_idx)

            results = classification_report(y_true, y_pred, digits=4)
            self.logger.info("***** Test results *****")
            self.logger.info("\n%s", results)  # 显示各类别结果
            f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])

            self.logger.info(f"--- Test f1 score is {f1_score} ---")
            # if errorArr:
            #     self.logger.info(f"本轮评估共有 {len(errorArr)} 个识别错误:")
            #     for error in errorArr:
            #         for key, value in error.items():
            #             self.logger.info(f"{key} : {value}\n")
    def _step(self, batch):
        input_ids, token_type_ids, attention_mask, image_features, labels, \
                exp_input_ids, exp_token_type_ids, exp_attention_mask,sentence,\
            img_exp_input_ids,img_exp_token_type_ids,img_exp_attention_mask,images_sim = batch

        # 修改模型调用，传入explaination参数
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            img_features=image_features,
            labels=labels,
            exp_input_ids=exp_input_ids,
            exp_token_type_ids=exp_token_type_ids,
            exp_attention_mask=exp_attention_mask,
            img_exp_input_ids=img_exp_input_ids,
            img_exp_token_type_ids=img_exp_token_type_ids,
            img_exp_attention_mask=img_exp_attention_mask,
            images_sim=images_sim
        )
        logits, loss = output
        return attention_mask, labels, logits, loss,sentence

    def bert_before_train(self):
        """创建优化器，支持BERT和调制器模块的不同学习率"""

        # ===== 获取分层学习率参数 =====
        bert_lr_scale = getattr(self.args, 'bert_lr_scale', 0.5)  # 默认0.5
        new_module_lr_scale = getattr(self.args, 'new_module_lr_scale', 5.0)  # 默认5.0

        bert_lr = self.args.lr * bert_lr_scale
        modulator_lr = self.args.lr * new_module_lr_scale

        # ===== 分离不同模块的参数 =====
        bert_params = []
        modulator_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # 根据参数名判断属于哪个模块
            if 'modulator' in name.lower():  # 调制器模块
                modulator_params.append(param)
            else:  # 其他都是BERT参数（包括bert.前缀的）
                bert_params.append(param)

        # ===== 创建参数组 =====
        optimizer_grouped_parameters = [
            {
                'params': bert_params,
                'lr': bert_lr,
                'weight_decay': 0.01
            },
            {
                'params': modulator_params,
                'lr': modulator_lr,
                'weight_decay': 0.01
            }
        ]

        # ===== 创建优化器 =====
        self.optimizer = optim.AdamW(optimizer_grouped_parameters)

        # 将模型移动到设备
        self.model.to(self.args.device)

        # ===== 学习率调度器（保持不变） =====
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * self.train_num_steps),
            num_training_steps=self.train_num_steps
        )
