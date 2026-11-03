import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.special import expit
from sklearn.metrics import multilabel_confusion_matrix

from tqdm import tqdm

from utils.training import classification_losses
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


def vali_classify(args, accelerator, model, vali_data, vali_loader, criterion):
    model.eval()
    total_loss, total_correct, total_labels, total_samples = 0.0, 0, 0, 0
    limit = args.debug_iters if getattr(args, "debug", False) else float("inf")
    with torch.no_grad():
        for index, batch in enumerate(vali_loader):
            if index >= limit:
                break
            signal = batch["data"].float().to(accelerator.device)
            labels = batch["label"].float().to(accelerator.device)
            use_amp = bool(args.use_amp and accelerator.device.type == "cuda")
            with torch.amp.autocast(device_type=accelerator.device.type, enabled=use_amp):
                outputs, global_logits, local_logits = model(signal)
                losses = classification_losses(outputs, global_logits, local_logits, labels, criterion,
                                               local_only=getattr(model, "local_only", False))
            batch_loss = losses["total"].detach().expand(len(labels))
            outputs, labels, batch_loss = accelerator.gather_for_metrics((outputs, labels, batch_loss))
            total_loss += batch_loss.sum().item()
            total_samples += len(labels)
            total_correct += ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
            total_labels += labels.numel()
    if not total_samples:
        raise ValueError("Validation split is empty")
    model.train()
    return total_loss / total_samples, total_correct / total_labels


def test_classify(args, accelerator, model, test_loader, criterion):
    model.eval()
    total_loss = []
    total_correct = 0
    total_labels = 0

    all_outputs = []
    all_preds = []
    all_labels = []


    weights = {
        "motor": 1.0,
        "gear": 1.0,
        "LA": 1.0,
        "RA": 0.5
    }
    max_iters = args.debug_iters if getattr(args, 'debug', False) else float('inf')


    num_batches_to_run = min(len(test_loader), max_iters) if getattr(args, 'debug', False) else len(test_loader)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader),
                             desc=" Testing",
                             total=num_batches_to_run,
                             disable=not accelerator.is_local_main_process):


            if i >= max_iters:
                break
            batch_x = batch['data'].float().to(accelerator.device)       # [B, seq_len, 21]
            batch_y = batch['label'].float().to(accelerator.device)      # [B, num_classes]


            final_logits, _, _ = model(batch_x)
            outputs = final_logits                                    # [B, num_classes]


            motor_out = outputs[:, 0:5]
            gear_out = outputs[:, 5:14]
            LA_out = outputs[:, 14:19]
            RA_out = outputs[:, 19:21]

            motor_y = batch_y[:, 0:5]
            gear_y = batch_y[:, 5:14]
            LA_y = batch_y[:, 14:19]
            RA_y = batch_y[:, 19:21]


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


            preds = (torch.sigmoid(outputs) > 0.5).float()


            outputs, batch_y, preds = accelerator.gather_for_metrics((outputs, batch_y, preds))


            correct = (preds == batch_y).float().sum().item()
            total_correct += correct
            total_labels += batch_y.numel()

            all_outputs.append(outputs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.cpu())


    avg_loss = np.mean(total_loss)
    label_wise_acc = total_correct / total_labels if total_labels > 0 else 0.0

    all_outputs = torch.cat(all_outputs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)


    if accelerator.is_local_main_process:
        print("\n Per-Head Test Performance Breakdown:")

        print(f"  Motor loss: {_select_criterion(criterion, 'motor')(all_outputs[:, 0:5], all_labels[:, 0:5]).item():.6f}")
        print(f"  Gear  loss: {_select_criterion(criterion, 'gear')(all_outputs[:, 5:14], all_labels[:, 5:14]).item():.6f}")
        print(f"  LA    loss: {_select_criterion(criterion, 'LA')(all_outputs[:, 14:19], all_labels[:, 14:19]).item():.6f}")
        print(f"  RA    loss: {_select_criterion(criterion, 'RA')(all_outputs[:, 19:21], all_labels[:, 19:21]).item():.6f}")

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

        print(f"  Motor metrics: {motor_metrics}")
        print(f"  Gear  metrics: {gear_metrics}")
        print(f"  LA    metrics: {LA_metrics}")
        print(f"  RA    metrics: {RA_metrics}")

        print(f"\n Overall Test Loss: {avg_loss:.6f} | label_wise_acc: {label_wise_acc:.6f}")

    model.train()
    return avg_loss, label_wise_acc, all_preds.numpy(), all_labels.numpy()


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
    """Evaluate lists of multilabel logits and targets at a fixed threshold."""

    all_preds = torch.cat(preds_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)


    all_probs = torch.sigmoid(all_preds)
    bin_preds = (all_probs > threshold).float()


    y_pred = bin_preds.cpu().numpy()
    y_true = all_labels.cpu().numpy()


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
        print("\n Evaluation Metrics Summary:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return metrics


def evaluate_metrics_extended(logits, labels, threshold=0.5, verbose=True, rare_label_indices=None):
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
        y_prob = expit(np.asarray(logits))
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

    rare_labels = [] if rare_label_indices is None else [int(i) for i in rare_label_indices]
    if any(i < 0 or i >= y_true.shape[1] for i in rare_labels):
        raise ValueError("Rare-label index is outside the label space")
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
        print("\n Extended Evaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")
    return metrics


def plot_convergence_curves(train_losses_steps, save_path, max_points_to_plot=2000):
    """Plot a smoothed, optionally downsampled training-loss curve."""
    print(f" Plotting training convergence curve ({len(train_losses_steps)} steps) and saving to {save_path}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    num_total_steps = len(train_losses_steps)
    if num_total_steps == 0:
        print(" Warning: No training loss data to plot.")
        plt.close(fig)
        return


    if num_total_steps > max_points_to_plot:

        factor = num_total_steps // max_points_to_plot


        losses_series = pd.Series(train_losses_steps)
        downsampled_losses = losses_series.rolling(window=factor, center=True).mean().dropna()

        steps_x = downsampled_losses.index.values
        losses_to_plot = downsampled_losses.values
        label_suffix = f" (downsampled from {num_total_steps} steps)"
    else:
        steps_x = np.arange(1, num_total_steps + 1)
        losses_to_plot = train_losses_steps
        label_suffix = " (per step)"


    ax.plot(steps_x, losses_to_plot, color='#1f77b4', label=f'Training Loss{label_suffix}', linewidth=2)


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
    print(f" Training convergence curve saved to {save_path}.")


def plot_co_occurrence_matrix(y_true, y_pred_binary, class_names, save_path, normalize=True):
    """Plot predicted-label co-occurrence conditioned on each true label."""
    print(f" Plotting label co-occurrence confusion matrix and saving to {save_path}...")
    num_classes = len(class_names)


    co_occurrence_matrix = np.zeros((num_classes, num_classes))


    for i in range(num_classes):


        true_i_indices = np.where(y_true[:, i] == 1)[0]


        if len(true_i_indices) == 0:
            continue


        pred_counts_for_true_i = y_pred_binary[true_i_indices].sum(axis=0)


        co_occurrence_matrix[i, :] = pred_counts_for_true_i


    if normalize:


        row_sums = y_true.sum(axis=0)

        row_sums[row_sums == 0] = 1

        normalized_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        matrix_to_plot = normalized_matrix
        fmt = '.2f'
        title = 'Normalized Label Co-occurrence Matrix (Recall on Diagonal)'
    else:
        matrix_to_plot = co_occurrence_matrix
        fmt = 'd'
        title = 'Label Co-occurrence Count Matrix'


    fig, ax = plt.subplots(figsize=(20, 16))

    sns.heatmap(matrix_to_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 8}, ax=ax)

    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(" Co-occurrence confusion matrix saved.")
