import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, average_precision_score

plt.switch_backend('agg')


def _select_criterion(criterion, name):
    if isinstance(criterion, dict):
        return criterion.get(name, criterion.get("full"))
    return criterion


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True, save_fn=None):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode
        self.save_fn = save_fn

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        checkpoint_path = path + '/' + 'checkpoint.pt'
        if self.save_fn is not None:
            self.save_fn(model, checkpoint_path)
        else:
            if self.accelerator is not None:
                model = self.accelerator.unwrap_model(model)
            if any(name.startswith("llm_model.") for name, _ in model.named_parameters()):
                raise RuntimeError(
                    "A checkpoint save callback is required for models containing a frozen LLM."
                )
            torch.save(model.state_dict(), checkpoint_path)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


# def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
#     total_loss = []
#     total_mae_loss = []
#     model.eval()
#     with torch.no_grad():
#         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
#             batch_x = batch_x.float().to(accelerator.device)
#             batch_y = batch_y.float()
#
#             batch_x_mark = batch_x_mark.float().to(accelerator.device)
#             batch_y_mark = batch_y_mark.float().to(accelerator.device)
#
#             # decoder input
#             dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
#             dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
#                 accelerator.device)
#             # encoder - decoder
#             if args.use_amp:
#                 with torch.cuda.amp.autocast():
#                     if args.output_attention:
#                         outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#             else:
#                 if args.output_attention:
#                     outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                 else:
#                     outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#             outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
#
#             f_dim = -1 if args.features == 'MS' else 0
#             outputs = outputs[:, -args.pred_len:, f_dim:]
#             batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
#
#             pred = outputs.detach()
#             true = batch_y.detach()
#
#             loss = criterion(pred, true)
#
#             mae_loss = mae_metric(pred, true)
#
#             total_loss.append(loss.item())
#             total_mae_loss.append(mae_loss.item())
#
#     total_loss = np.average(total_loss)
#     total_mae_loss = np.average(total_mae_loss)
#
#     model.train()
#     return total_loss, total_mae_loss
def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    model.eval()
    total_loss = []
    total_mae_loss = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader), disable=not accelerator.is_local_main_process):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # 构造 decoder 输入
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(accelerator.device)

            # 模型前向计算
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if args.output_attention:
                        outputs = outputs[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if args.output_attention:
                    outputs = outputs[0]

            # 分布式环境下收集所有进程的输出和标签
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            # 提取预测长度部分，并调整特征维度
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            # 计算损失和MAE
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            mae = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae.item())

    # 汇总平均损失
    avg_loss = np.mean(total_loss)
    avg_mae = np.mean(total_mae_loss)

    model.train()
    return avg_loss, avg_mae

def vali_classify(args, accelerator, model, vali_data, vali_loader, criterion):
    model.eval()
    total_loss = []
    total_correct = 0
    total_labels = 0
    MAX_ITERS_PER_EPOCH = args.debug_iters if getattr(args, "debug", False) else float('inf')
    all_preds = []
    all_labels = []

    # 各头权重（与训练阶段保持一致）
    weights = {
        "motor": 1.0,
        "gear": 1.0,
        "LA": 1.0,
        "RA": 0.5
    }

    with torch.no_grad():
        for i, batch in tqdm(enumerate(vali_loader), desc="Valing", disable=not accelerator.is_local_main_process):
            if i >= MAX_ITERS_PER_EPOCH:
                break

            batch_x = batch['data'].float().to(accelerator.device)     # [B, seq_len, 21]
            batch_y = batch['label'].float()                            # [B, num_classes]

            # 模型前向计算
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    final_logits, _, _ = model(batch_x)
                    outputs = final_logits                          # [B, num_classes]
            else:
                final_logits, _, _ = model(batch_x)
                outputs = final_logits

            # 分布式聚合所有预测与标签
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
            batch_y = batch_y.to(accelerator.device)

            # === 拆分输出与标签 ===
            motor_out = outputs[:, 0:5]
            gear_out = outputs[:, 5:14]
            LA_out = outputs[:, 14:19]
            RA_out = outputs[:, 19:21]

            motor_y = batch_y[:, 0:5]
            gear_y = batch_y[:, 5:14]
            LA_y = batch_y[:, 14:19]
            RA_y = batch_y[:, 19:21]

            # === 计算各头 loss ===
            motor_loss = _select_criterion(criterion, "motor")(motor_out, motor_y)
            gear_loss = _select_criterion(criterion, "gear")(gear_out, gear_y)
            LA_loss = _select_criterion(criterion, "LA")(LA_out, LA_y)
            RA_loss = _select_criterion(criterion, "RA")(RA_out, RA_y)

            # === 按权重加权合并 ===
            weighted_loss = (
                weights["motor"] * motor_loss +
                weights["gear"] * gear_loss +
                weights["LA"] * LA_loss +
                weights["RA"] * RA_loss
            )
            loss = weighted_loss

            total_loss.append(loss.item())

            # === 多标签预测准确率 ===
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct = (preds == batch_y).float().sum().item()
            total_correct += correct
            total_labels += batch_y.numel()

            all_preds.append(outputs.detach().cpu())
            all_labels.append(batch_y.detach().cpu())

    # === 汇总 ===
    avg_loss = np.mean(total_loss)
    label_wise_acc = total_correct / total_labels if total_labels > 0 else 0.0

    # === 计算更详细的指标（精确率、召回率、F1 等） ===
    metrics = evaluate_metrics(all_preds, all_labels)

    # 输出可读的验证信息
    if accelerator.is_local_main_process:
        print(f"🧩 Validation Head Loss Breakdown:")
        print(
            f"  ┣━ Motor loss (mean): {np.mean([_select_criterion(criterion, 'motor')(o[:, 0:5], y[:, 0:5]).item() for o, y in zip(all_preds, all_labels)]):.6f}")
        print(
            f"  ┣━ Gear  loss (mean): {np.mean([_select_criterion(criterion, 'gear')(o[:, 5:14], y[:, 5:14]).item() for o, y in zip(all_preds, all_labels)]):.6f}")
        print(
            f"  ┣━ LA    loss (mean): {np.mean([_select_criterion(criterion, 'LA')(o[:, 14:19], y[:, 14:19]).item() for o, y in zip(all_preds, all_labels)]):.6f}")
        print(
            f"  ┗━ RA    loss (mean): {np.mean([_select_criterion(criterion, 'RA')(o[:, 19:21], y[:, 19:21]).item() for o, y in zip(all_preds, all_labels)]):.6f}")

    model.train()
    return avg_loss, label_wise_acc


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss

def test_classify(args, accelerator, model, test_loader, criterion):
    model.eval()
    total_loss = []
    total_correct = 0
    total_labels = 0

    all_outputs = []
    all_preds = []
    all_labels = []

    # 各头权重（保持与训练一致）
    weights = {
        "motor": 1.0,
        "gear": 1.0,
        "LA": 1.0,
        "RA": 0.5
    }
    max_iters = args.debug_iters if getattr(args, 'debug', False) else float('inf')

    # (可选但推荐) 动态设置 tqdm 的总长度，使进度条在 debug 模式下能正确显示
    num_batches_to_run = min(len(test_loader), max_iters) if getattr(args, 'debug', False) else len(test_loader)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader),
                             desc="🔍 Testing",
                             total=num_batches_to_run,  # 应用动态总长度
                             disable=not accelerator.is_local_main_process):

            # 如果当前迭代次数达到了 debug 模式的上限，则提前中断循环
            if i >= max_iters:
                break
            batch_x = batch['data'].float().to(accelerator.device)       # [B, seq_len, 21]
            batch_y = batch['label'].float().to(accelerator.device)      # [B, num_classes]

            # === 模型前向 ===
            final_logits, _, _ = model(batch_x)
            outputs = final_logits                                    # [B, num_classes]

            # === 拆分输出与标签 ===
            motor_out = outputs[:, 0:5]
            gear_out = outputs[:, 5:14]
            LA_out = outputs[:, 14:19]
            RA_out = outputs[:, 19:21]

            motor_y = batch_y[:, 0:5]
            gear_y = batch_y[:, 5:14]
            LA_y = batch_y[:, 14:19]
            RA_y = batch_y[:, 19:21]

            # === 分头计算 loss ===
            motor_loss = _select_criterion(criterion, "motor")(motor_out, motor_y)
            gear_loss = _select_criterion(criterion, "gear")(gear_out, gear_y)
            LA_loss = _select_criterion(criterion, "LA")(LA_out, LA_y)
            RA_loss = _select_criterion(criterion, "RA")(RA_out, RA_y)

            weighted_loss = (
                weights["motor"] * motor_loss +
                weights["gear"] * gear_loss +
                weights["LA"] * LA_loss +
                weights["RA"] * RA_loss
            )
            loss = weighted_loss / sum(weights.values())
            total_loss.append(loss.item())

            # === 多标签预测 ===
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # 分布式聚合结果
            outputs, batch_y, preds = accelerator.gather_for_metrics((outputs, batch_y, preds))

            # === 正确率 ===
            correct = (preds == batch_y).float().sum().item()
            total_correct += correct
            total_labels += batch_y.numel()

            all_outputs.append(outputs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.cpu())

    # === 汇总指标 ===
    avg_loss = np.mean(total_loss)
    label_wise_acc = total_correct / total_labels if total_labels > 0 else 0.0

    all_outputs = torch.cat(all_outputs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # === 打印分头指标 ===
    if accelerator.is_local_main_process:
        print("\n📊 Per-Head Test Performance Breakdown:")

        print(f"  ┣━ Motor loss: {_select_criterion(criterion, 'motor')(all_outputs[:, 0:5], all_labels[:, 0:5]).item():.6f}")
        print(f"  ┣━ Gear  loss: {_select_criterion(criterion, 'gear')(all_outputs[:, 5:14], all_labels[:, 5:14]).item():.6f}")
        print(f"  ┣━ LA    loss: {_select_criterion(criterion, 'LA')(all_outputs[:, 14:19], all_labels[:, 14:19]).item():.6f}")
        print(f"  ┗━ RA    loss: {_select_criterion(criterion, 'RA')(all_outputs[:, 19:21], all_labels[:, 19:21]).item():.6f}")

        motor_metrics = evaluate_metrics(
            [all_outputs[:, 0:5]],
            [all_labels[:, 0:5]],
            threshold=0.5, verbose=False
        )
        gear_metrics = evaluate_metrics(
            [all_outputs[:, 5:14]],
            [all_labels[:, 5:14]],
            threshold=0.5, verbose=False
        )
        LA_metrics = evaluate_metrics(
            [all_outputs[:, 14:19]],
            [all_labels[:, 14:19]],
            threshold=0.5, verbose=False
        )
        RA_metrics = evaluate_metrics(
            [all_outputs[:, 19:21]],
            [all_labels[:, 19:21]],
            threshold=0.5, verbose=False
        )

        print(f"  ┣━ Motor metrics: {motor_metrics}")
        print(f"  ┣━ Gear  metrics: {gear_metrics}")
        print(f"  ┣━ LA    metrics: {LA_metrics}")
        print(f"  ┗━ RA    metrics: {RA_metrics}")

        print(f"\n✅ Overall Test Loss: {avg_loss:.6f} | label_wise_acc: {label_wise_acc:.6f}")

    model.train()
    return avg_loss, label_wise_acc, all_preds.numpy(), all_labels.numpy()



def shapeformerTest(args, accelerator, model, test_loader, criterion):
    """
    更新后的验证函数，用于评估多任务分类模型。
    该函数现在返回 test_loss, test_acc, all_preds, all_labels。

    Args:
        args: 命令行参数或配置对象。
        accelerator: Accelerate 库的加速器对象。
        model: 待评估的模型。
        test_loader: 测试数据加载器。
        criterion: 损失函数 (例如 nn.BCEWithLogitsLoss)。

    Returns:
        tuple: 包含四个元素的元组
            - test_loss (float): 平均测试损失。
            - test_acc (float): 标签级的平均测试准确率。
            - all_preds (np.ndarray): 所有样本的预测结果 (numpy 数组)。
            - all_labels (np.ndarray): 所有样本的真实标签 (numpy 数组)。
    """
    model.eval()

    total_loss = []
    all_preds_list = []
    all_labels_list = []

    # 各任务权重
    weights = {"motor": 1.0, "gear": 1.0, "LA": 1.0, "RA": 0.5}
    weight_sum = sum(weights.values())

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🔍 Validating", disable=not accelerator.is_local_main_process):
            # 1. 将数据移动到正确的设备
            batch_x = batch['data'].float().to(accelerator.device)
            batch_y = batch['label'].float().to(accelerator.device)

            # 2. 模型前向传播
            logits = model(batch_x)

            # 3. 拆分 logits 和 labels 以计算分头损失
            motor_out, gear_out, LA_out, RA_out = logits[:, 0:5], logits[:, 5:14], logits[:, 14:19], logits[:, 19:21]
            motor_y, gear_y, LA_y, RA_y = batch_y[:, 0:5], batch_y[:, 5:14], batch_y[:, 14:19], batch_y[:, 19:21]

            # 4. 计算加权总损失
            motor_loss = _select_criterion(criterion, "motor")(motor_out, motor_y)
            gear_loss = _select_criterion(criterion, "gear")(gear_out, gear_y)
            LA_loss = _select_criterion(criterion, "LA")(LA_out, LA_y)
            RA_loss = _select_criterion(criterion, "RA")(RA_out, RA_y)

            loss = (weights["motor"] * motor_loss +
                    weights["gear"] * gear_loss +
                    weights["LA"] * LA_loss +
                    weights["RA"] * RA_loss) / weight_sum

            total_loss.append(loss.item())

            # 5. 计算预测结果
            preds = (torch.sigmoid(logits) > 0.5).float()

            # 6. 聚合分布式环境下的结果
            preds, batch_y = accelerator.gather_for_metrics((preds, batch_y))

            all_preds_list.append(preds.cpu())
            all_labels_list.append(batch_y.cpu())

    # 将所有批次的结果拼接成一个大张量
    all_preds_tensor = torch.cat(all_preds_list, dim=0)
    all_labels_tensor = torch.cat(all_labels_list, dim=0)

    # === 计算最终指标 ===
    test_loss = np.mean(total_loss)
    label_wise_acc = (all_preds_tensor == all_labels_tensor).float().mean().item()

    # 将最终结果转换为 NumPy 数组
    all_preds = all_preds_tensor.numpy()
    all_labels = all_labels_tensor.numpy()

    # (可选) 仍然在主进程打印详细信息
    if accelerator.is_local_main_process:
        print(f"\n✅ Validation Complete:")
        print(f"   ┣━ Average Loss (test_loss): {test_loss:.6f}")
        print(f"   ┗━ label_wise_acc: {label_wise_acc:.4%}")
        # ... (可以保留或删除分任务的打印) ...

    # 确保在验证结束后将模型切换回训练模式
    model.train()

    # === 按照您的要求返回四个变量 ===
    return test_loss, label_wise_acc, all_preds, all_labels


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    prompt_path = './dataset/prompt_bank/{0}.txt'.format(file)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            content = f.read()
        return content
    return getattr(args, "content", "")


def evaluate_metrics(preds_list, labels_list, threshold=0.5, verbose=True):
    """
    评估多标签分类任务的常见指标。

    参数：
        preds_list: List of predicted logits (before sigmoid), shape: [B, C]
        labels_list: List of ground-truth labels, shape: [B, C]
        threshold: float, 阈值用于将 sigmoid 后的概率转换为 0/1
        verbose: 是否打印详细指标报告

    返回：
        metrics: dict，包含 f1, precision, recall, accuracy 等结果
    """
    # 拼接所有 batch
    all_preds = torch.cat(preds_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # sigmoid + 二值化
    all_probs = torch.sigmoid(all_preds)
    bin_preds = (all_probs > threshold).float()

    # 转 numpy
    y_pred = bin_preds.cpu().numpy()
    y_true = all_labels.cpu().numpy()

    # 计算指标
    label_wise_acc = float(np.mean(y_true == y_pred))
    metrics = {
        'label_wise_acc': label_wise_acc,
        'Hamming_Accuracy': label_wise_acc,
        'EMR': float(np.mean(np.all(y_true == y_pred, axis=1))),
        'Exact_match_ratio': float(np.mean(np.all(y_true == y_pred, axis=1))),
        'Micro_F1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'Macro_F1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_precision': float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_recall': float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        # Backward-compatible aliases.
        'F1_micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'F1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'Precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'Recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'Hamming_accuracy': label_wise_acc,
        'Sample_wise_accuracy': label_wise_acc,
    }

    if verbose:
        print("\n📊 Evaluation Metrics Summary:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return metrics


def evaluate_metrics_extended(logits, labels, threshold=0.5, verbose=True):
    """
    Extended multilabel metrics for PHM ablations.

    Component ranges follow Dataset_PHM label order:
    motor 0:5, gear 5:14, LA 14:19, RA 19:21.
    """
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    if isinstance(logits, torch.Tensor):
        y_prob = torch.sigmoid(logits).detach().cpu().numpy()
    else:
        y_prob = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    if isinstance(labels, torch.Tensor):
        y_true = labels.detach().cpu().numpy()
    else:
        y_true = np.asarray(labels)
    y_pred = (y_prob > threshold).astype(np.float32)

    metrics = {
        "label_wise_acc": float(np.mean(y_true == y_pred)),
        "EMR": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "Exact_match_ratio": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "Hamming_Accuracy": float(1.0 - np.mean(np.not_equal(y_true, y_pred))),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Micro_F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "sklearn_zero_division": 0,
    }
    component_ranges = {
        "motor": (0, 5),
        "gear": (5, 14),
        "LA": (14, 19),
        "RA": (19, 21),
    }
    metrics["per_component_Macro_F1"] = {}
    for name, (start, end) in component_ranges.items():
        metrics["per_component_Macro_F1"][name] = float(
            f1_score(y_true[:, start:end], y_pred[:, start:end], average="macro", zero_division=0)
        )

    positive_counts = y_true.sum(axis=0)
    positive_labels = np.where(positive_counts > 0)[0]
    if len(positive_labels) > 0:
        rare_cutoff = np.percentile(positive_counts[positive_labels], 25)
        rare_labels = [int(i) for i in positive_labels if positive_counts[i] <= rare_cutoff]
    else:
        rare_labels = []
    if rare_labels:
        metrics["rare_label_indices"] = rare_labels
        metrics["rare_label_F1"] = float(
            f1_score(y_true[:, rare_labels], y_pred[:, rare_labels], average="macro", zero_division=0)
        )
    else:
        metrics["rare_label_indices"] = []
        metrics["rare_label_F1"] = None

    auprc = {}
    valid_scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            auprc[str(i)] = None
            continue
        score = float(average_precision_score(y_true[:, i], y_prob[:, i]))
        auprc[str(i)] = score
        valid_scores.append(score)
    metrics["positive_label_AUPRC"] = auprc
    metrics["Macro_AUPRC_valid_labels"] = float(np.mean(valid_scores)) if valid_scores else None
    metrics["AUPRC"] = metrics["Macro_AUPRC_valid_labels"]

    if verbose:
        print("\n📊 Extended Evaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")
    return metrics


import torch
import numpy as np
from tqdm import tqdm


def timeMIL_vail(args, model, vali_loader, criterion, accelerator):
    """
    Validation function adapted for a single-output model like TimeMIL.
    Handles input permutation, simplified loss calculation, and metrics.
    """
    model.eval()
    total_loss = []
    total_correct = 0
    total_samples = 0
    MAX_ITERS_PER_EPOCH = args.debug_iters if args.debug else float('inf')

    # <--- MODIFIED: 移除 all_preds 和 all_labels，因为此函数只返回 loss 和 acc
    # 如果需要详细指标，可以在 test_classify 中实现

    # <--- MODIFIED: 移除了多头权重字典

    with torch.no_grad():
        for i, batch in tqdm(enumerate(vali_loader), desc="[Validating]", total=len(vali_loader)):
            if i >= MAX_ITERS_PER_EPOCH:
                break

            batch_x = batch['data'].float().to(accelerator.device)
            batch_y = batch['label'].float().to(accelerator.device)

            # <--- MODIFIED: 核心改动！调整输入维度以匹配 TimeMIL ---
            # [B, 21, seq_len] -> [B, seq_len, 21]
            batch_x = batch_x.permute(0, 2, 1)

            # <--- MODIFIED: 简化的模型前向传播 ---
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)  # TimeMIL 直接返回 [B, 21] 的 logits
            else:
                outputs = model(batch_x)

            # <--- MODIFIED: 简化的损失计算 ---
            loss = criterion(outputs, batch_y)

            # 使用 accelerator.gather_for_metrics 聚合所有设备的结果
            gathered_outputs, gathered_labels = accelerator.gather_for_metrics((outputs, batch_y))

            # 在主进程上计算指标
            if accelerator.is_main_process:
                total_loss.append(loss.item())

                # === 多标签预测准确率 ===
                preds = (torch.sigmoid(gathered_outputs) > 0.5).float()
                correct = (preds == gathered_labels).float().sum().item()
                total_correct += correct
                total_samples += gathered_labels.numel()

    # 在所有进程上同步，确保主进程完成所有计算
    accelerator.wait_for_everyone()

    # 主进程返回最终结果
    if accelerator.is_main_process:
        avg_loss = np.mean(total_loss)
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    else:
        # 非主进程可以返回默认值
        avg_loss, avg_acc = 0.0, 0.0

    model.train()  # 恢复模型到训练模式

    return avg_loss, avg_acc


import torch
import numpy as np
from tqdm import tqdm


def unified_test_classify(args, model, test_loader, criterion, accelerator):
    """
    一个统一的、能适配所有对比模型的测试函数。
    它会根据 args.model 来决定是否需要对输入数据进行维度转置。
    """
    model.eval()
    total_loss = []
    all_preds = []
    all_labels = []

    max_iters = args.debug_iters if args.debug else float('inf')
    num_batches_to_run = min(len(test_loader), max_iters) if args.debug else len(test_loader)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader),
                             desc="🔍 Unified Testing",
                             total=num_batches_to_run,
                             disable=not accelerator.is_local_main_process):

            if i >= max_iters:
                break

            batch_x = batch['data'].float().to(accelerator.device)
            batch_y = batch['label'].float().to(accelerator.device)

            # ======================== 核心修改 ========================
            # 根据模型名称，决定是否对输入进行 permute
            if args.model in ['TimeMIL']: # 假设只有 TimeMIL 需要 [B, C, T]
                batch_x = batch_x.permute(0, 2, 1)
            # Sundial, Timer, and KG-LLMoE expect [B, T, C], so no permutation is needed.
            # ==========================================================

            # 使用 AMP (Automatic Mixed Precision)
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    # 统一的前向传播调用
                    outputs, _, _ = model(batch_x)
                    loss = criterion(outputs, batch_y)
            else:
                outputs, _, _ = model(batch_x)
                loss = criterion(outputs, batch_y)

            # --- 后续的 gather, loss/acc 计算等逻辑保持不变 ---
            gathered_outputs, gathered_labels, gathered_loss = accelerator.gather_for_metrics(
                (outputs, batch_y, loss)
            )

            if accelerator.is_main_process:
                if gathered_loss.dim() > 0:
                    total_loss.extend(gathered_loss.cpu().numpy().tolist())
                else:
                    total_loss.append(gathered_loss.item())
                all_preds.append(gathered_outputs.cpu())
                all_labels.append(gathered_labels.cpu())

    accelerator.wait_for_everyone()

    avg_loss, avg_acc = 0.0, 0.0
    final_preds_np, final_labels_np = np.array([]), np.array([])

    if accelerator.is_main_process:
        # --- MODIFICATION START ---
        # 6. 添加一个安全检查，防止在0迭代时出错
        if not all_preds:
            print("⚠️ Warning: No batches were processed in the test loop (check debug settings).")
            return avg_loss, avg_acc, final_preds_np, final_labels_np
        # --- MODIFICATION END ---

        avg_loss = np.mean(total_loss)
        final_preds = torch.cat(all_preds, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        preds_binary = (torch.sigmoid(final_preds) > 0.5).float()
        total_correct = (preds_binary == final_labels).float().sum().item()
        total_samples = final_labels.numel()
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        final_preds_np = final_preds.numpy()
        final_labels_np = final_labels.numpy()

    return avg_loss, avg_acc, final_preds_np, final_labels_np


def plot_convergence_curves(train_losses_steps, save_path, max_points_to_plot=2000):
    """
    绘制并保存训练损失的平滑收敛曲线，并能处理海量数据点。

    Args:
        train_losses_steps (list): 每个step的训练损失列表。
        save_path (str): 图片的保存路径。
        max_points_to_plot (int): 图上最多绘制的点数，用于降采样。
    """
    print(f"📈 Plotting training convergence curve ({len(train_losses_steps)} steps) and saving to {save_path}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    num_total_steps = len(train_losses_steps)
    if num_total_steps == 0:
        print("⚠️ Warning: No training loss data to plot.")
        plt.close(fig)
        return

    # --- 1. 智能降采样 ---
    if num_total_steps > max_points_to_plot:
        # 计算降采样因子
        factor = num_total_steps // max_points_to_plot
        # 使用 pandas 的 rolling().mean() 来进行高效的滚动平均，实现降采样
        # 这比手动的循环快得多
        losses_series = pd.Series(train_losses_steps)
        downsampled_losses = losses_series.rolling(window=factor, center=True).mean().dropna()

        steps_x = downsampled_losses.index.values
        losses_to_plot = downsampled_losses.values
        label_suffix = f" (downsampled from {num_total_steps} steps)"
    else:
        steps_x = np.arange(1, num_total_steps + 1)
        losses_to_plot = train_losses_steps
        label_suffix = " (per step)"

    # --- 2. 绘图 ---
    ax.plot(steps_x, losses_to_plot, color='#1f77b4', label=f'Training Loss{label_suffix}', linewidth=2)

    # --- 3. 美化图表 ---
    ax.set_title('Training Loss Convergence Curve', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    min_loss, max_loss = min(losses_to_plot), max(losses_to_plot)
    ax.set_ylim(bottom=min_loss * 0.9, top=max_loss * 1.1)
    ax.set_xlim(0, num_total_steps)

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Training convergence curve saved to {save_path}.")


def plot_co_occurrence_matrix(y_true, y_pred_binary, class_names, save_path, normalize=True):
    """
    计算并绘制多标签任务中的“标签共现混淆矩阵”。
    矩阵的(i, j)格表示，在所有真实标签为i的样本中，预测标签为j的比例。

    Args:
        y_true (np.array): 真实标签, shape [N, C].
        y_pred_binary (np.array): 二值化后的预测标签, shape [N, C].
        class_names (list): 类别名称列表, length C.
        save_path (str): 图片的保存路径。
        normalize (bool): 是否对行进行归一化，显示为百分比。
    """
    print(f"📈 Plotting label co-occurrence confusion matrix and saving to {save_path}...")
    num_classes = len(class_names)

    # 初始化一个 C x C 的共现矩阵
    co_occurrence_matrix = np.zeros((num_classes, num_classes))

    # 遍历每一个类别 i (作为真实发生的故障)
    for i in range(num_classes):
        # 找到所有真实标签为 i 的样本的索引
        # y_true[:, i] == 1 会返回一个布尔掩码
        true_i_indices = np.where(y_true[:, i] == 1)[0]

        # 如果这个故障从未在数据集中出现过，则跳过
        if len(true_i_indices) == 0:
            continue

        # 在这些样本中，查看其他所有类别 j 的预测情况
        # y_pred_binary[true_i_indices] 是一个子集
        # .sum(axis=0) 会对这个子集的所有样本，按列（即按类别）求和
        # 结果是一个长度为 C 的向量，其第j个元素表示，在i发生时，j被预测为1的总次数
        pred_counts_for_true_i = y_pred_binary[true_i_indices].sum(axis=0)

        # 填入共现矩阵的第 i 行
        co_occurrence_matrix[i, :] = pred_counts_for_true_i

    # --- 归一化 (非常重要) ---
    if normalize:
        # 我们对每一行进行归一化，除以该行对应的真实故障的总发生次数
        # co_occurrence_matrix.sum(axis=1, keepdims=True) 会得到每行真实故障的总数
        # 这里更准确的应该是除以 y_true[:, i].sum()
        row_sums = y_true.sum(axis=0)
        # 防止除以零
        row_sums[row_sums == 0] = 1
        # 使用广播机制进行除法
        normalized_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        matrix_to_plot = normalized_matrix
        fmt = '.2f'  # 显示为两位小数
        title = 'Normalized Label Co-occurrence Matrix (Recall on Diagonal)'
    else:
        matrix_to_plot = co_occurrence_matrix
        fmt = 'd'  # 显示为整数
        title = 'Label Co-occurrence Count Matrix'

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(20, 16))  # 使用一个更大的画布

    sns.heatmap(matrix_to_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 8}, ax=ax)

    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)
    plt.xticks(rotation=45, ha='right')  # 旋转X轴标签，防止重叠
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print("✅ Co-occurrence confusion matrix saved.")
