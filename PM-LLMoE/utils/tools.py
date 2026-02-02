import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    lr_adjust = {}
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
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

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
            msg = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...'
            if self.accelerator is not None:
                self.accelerator.print(msg)
            else:
                print(msg)

        checkpoint_path = path + '/' + 'checkpoint.pt'
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), checkpoint_path)
        else:
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


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    model.eval()
    total_loss = []
    total_mae_loss = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader),
                                                                      disable=not accelerator.is_local_main_process):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Construct decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(accelerator.device)

            # Model forward
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if args.output_attention:
                        outputs = outputs[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if args.output_attention:
                    outputs = outputs[0]

            # Gather outputs and labels in distributed environment
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            # Extract prediction length and adjust feature dimensions
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            # Calculate Loss and MAE
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            mae = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae.item())

    # Aggregate average loss
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

    # Head weights (consistent with training stage)
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

            batch_x = batch['data'].float().to(accelerator.device)  # [B, seq_len, 21]
            batch_y = batch['label'].float()  # [B, num_classes]

            # Model forward
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    final_logits, _, _ = model(batch_x)
                    outputs = final_logits  # [B, num_classes]
            else:
                final_logits, _, _ = model(batch_x)
                outputs = final_logits

            # Distributed aggregation
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
            batch_y = batch_y.to(accelerator.device)

            # === Split outputs and labels ===
            motor_out = outputs[:, 0:5]
            gear_out = outputs[:, 5:14]
            LA_out = outputs[:, 14:19]
            RA_out = outputs[:, 19:21]

            motor_y = batch_y[:, 0:5]
            gear_y = batch_y[:, 5:14]
            LA_y = batch_y[:, 14:19]
            RA_y = batch_y[:, 19:21]

            # === Calculate head loss ===
            motor_loss = criterion(motor_out, motor_y)
            gear_loss = criterion(gear_out, gear_y)
            LA_loss = criterion(LA_out, LA_y)
            RA_loss = criterion(RA_out, RA_y)

            # === Weighted combination ===
            weighted_loss = (
                    weights["motor"] * motor_loss +
                    weights["gear"] * gear_loss +
                    weights["LA"] * LA_loss +
                    weights["RA"] * RA_loss
            )
            loss = weighted_loss

            total_loss.append(loss.item())

            # === Multi-label prediction accuracy ===
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct = (preds == batch_y).float().sum().item()
            total_correct += correct
            total_labels += batch_y.numel()

            all_preds.append(outputs.detach().cpu())
            all_labels.append(batch_y.detach().cpu())

    # === Summary ===
    avg_loss = np.mean(total_loss)
    avg_acc = total_correct / total_labels if total_labels > 0 else 0.0

    # === Calculate detailed metrics ===
    metrics = evaluate_metrics(all_preds, all_labels, verbose=False)

    # Output readable validation info
    if accelerator.is_local_main_process:
        print(f"🧩 Validation Head Loss Breakdown:")
        print(
            f"  ┣━ Motor loss (mean): {np.mean([criterion(o[:, 0:5], y[:, 0:5]).item() for o, y in zip(all_preds, all_labels)]):.6f}")
        print(
            f"  ┣━ Gear  loss (mean): {np.mean([criterion(o[:, 5:14], y[:, 5:14]).item() for o, y in zip(all_preds, all_labels)]):.6f}")
        print(
            f"  ┣━ LA    loss (mean): {np.mean([criterion(o[:, 14:19], y[:, 14:19]).item() for o, y in zip(all_preds, all_labels)]):.6f}")
        print(
            f"  ┗━ RA    loss (mean): {np.mean([criterion(o[:, 19:21], y[:, 19:21]).item() for o, y in zip(all_preds, all_labels)]):.6f}")

    model.train()
    return avg_loss, avg_acc


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

    # Head weights (keep consistent with training)
    weights = {
        "motor": 1.0,
        "gear": 1.0,
        "LA": 1.0,
        "RA": 0.5
    }
    max_iters = args.debug_iters if getattr(args, 'debug', False) else float('inf')

    # Dynamically set tqdm length
    num_batches_to_run = min(len(test_loader), max_iters) if getattr(args, 'debug', False) else len(test_loader)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader),
                             desc="🔍 Testing",
                             total=num_batches_to_run,
                             disable=not accelerator.is_local_main_process):

            if i >= max_iters:
                break
            batch_x = batch['data'].float().to(accelerator.device)  # [B, seq_len, 21]
            batch_y = batch['label'].float().to(accelerator.device)  # [B, num_classes]

            # === Model Forward ===
            final_logits, _, _ = model(batch_x)
            outputs = final_logits  # [B, num_classes]

            # === Split Outputs and Labels ===
            motor_out = outputs[:, 0:5]
            gear_out = outputs[:, 5:14]
            LA_out = outputs[:, 14:19]
            RA_out = outputs[:, 19:21]

            motor_y = batch_y[:, 0:5]
            gear_y = batch_y[:, 5:14]
            LA_y = batch_y[:, 14:19]
            RA_y = batch_y[:, 19:21]

            # === Calculate Head Loss ===
            motor_loss = criterion(motor_out, motor_y)
            gear_loss = criterion(gear_out, gear_y)
            LA_loss = criterion(LA_out, LA_y)
            RA_loss = criterion(RA_out, RA_y)

            weighted_loss = (
                    weights["motor"] * motor_loss +
                    weights["gear"] * gear_loss +
                    weights["LA"] * LA_loss +
                    weights["RA"] * RA_loss
            )
            loss = weighted_loss / sum(weights.values())
            total_loss.append(loss.item())

            # === Multi-label Prediction ===
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Distributed aggregation
            outputs, batch_y, preds = accelerator.gather_for_metrics((outputs, batch_y, preds))

            # === Accuracy ===
            correct = (preds == batch_y).float().sum().item()
            total_correct += correct
            total_labels += batch_y.numel()

            all_outputs.append(outputs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.cpu())

    # === Metrics Summary ===
    avg_loss = np.mean(total_loss)
    accuracy = total_correct / total_labels if total_labels > 0 else 0.0

    all_outputs = torch.cat(all_outputs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # === Print Head Metrics ===
    if accelerator.is_local_main_process:
        print("\n📊 Per-Head Test Performance Breakdown:")

        print(f"  ┣━ Motor loss: {criterion(all_outputs[:, 0:5], all_labels[:, 0:5]).item():.6f}")
        print(f"  ┣━ Gear  loss: {criterion(all_outputs[:, 5:14], all_labels[:, 5:14]).item():.6f}")
        print(f"  ┣━ LA    loss: {criterion(all_outputs[:, 14:19], all_labels[:, 14:19]).item():.6f}")
        print(f"  ┗━ RA    loss: {criterion(all_outputs[:, 19:21], all_labels[:, 19:21]).item():.6f}")

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

        print(f"\n✅ Overall Test Loss: {avg_loss:.6f} | Accuracy: {accuracy:.6f}")

    model.train()
    return avg_loss, accuracy, all_preds.numpy(), all_labels.numpy()


def shapeformerTest(args, accelerator, model, test_loader, criterion):
    """
    Updated validation function for multi-task classification models.
    Returns: test_loss, test_acc, all_preds, all_labels.
    """
    model.eval()

    total_loss = []
    all_preds_list = []
    all_labels_list = []

    # Task weights
    weights = {"motor": 1.0, "gear": 1.0, "LA": 1.0, "RA": 0.5}
    weight_sum = sum(weights.values())

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🔍 Validating", disable=not accelerator.is_local_main_process):
            # 1. Move data to device
            batch_x = batch['data'].float().to(accelerator.device)
            batch_y = batch['label'].float().to(accelerator.device)

            # 2. Model forward
            logits = model(batch_x)

            # 3. Split logits and labels
            motor_out, gear_out, LA_out, RA_out = logits[:, 0:5], logits[:, 5:14], logits[:, 14:19], logits[:, 19:21]
            motor_y, gear_y, LA_y, RA_y = batch_y[:, 0:5], batch_y[:, 5:14], batch_y[:, 14:19], batch_y[:, 19:21]

            # 4. Calculate weighted loss
            motor_loss = criterion(motor_out, motor_y)
            gear_loss = criterion(gear_out, gear_y)
            LA_loss = criterion(LA_out, LA_y)
            RA_loss = criterion(RA_out, RA_y)

            loss = (weights["motor"] * motor_loss +
                    weights["gear"] * gear_loss +
                    weights["LA"] * LA_loss +
                    weights["RA"] * RA_loss) / weight_sum

            total_loss.append(loss.item())

            # 5. Calculate predictions
            preds = (torch.sigmoid(logits) > 0.5).float()

            # 6. Aggregate results
            preds, batch_y = accelerator.gather_for_metrics((preds, batch_y))

            all_preds_list.append(preds.cpu())
            all_labels_list.append(batch_y.cpu())

    # Concatenate results
    all_preds_tensor = torch.cat(all_preds_list, dim=0)
    all_labels_tensor = torch.cat(all_labels_list, dim=0)

    # === Calculate Final Metrics ===
    test_loss = np.mean(total_loss)
    test_acc = (all_preds_tensor == all_labels_tensor).float().mean().item()

    # Convert to numpy
    all_preds = all_preds_tensor.numpy()
    all_labels = all_labels_tensor.numpy()

    # Print details in main process
    if accelerator.is_local_main_process:
        print(f"\n✅ Validation Complete:")
        print(f"   ┣━ Average Loss (test_loss): {test_loss:.6f}")
        print(f"   ┗━ Label-wise Accuracy (test_acc): {test_acc:.4%}")

    model.train()

    return test_loss, test_acc, all_preds, all_labels


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content


def evaluate_metrics(preds_list, labels_list, threshold=0.5, verbose=True):
    """
    Evaluates common multi-label classification metrics.

    Args:
        preds_list: List of predicted logits (before sigmoid), shape: [B, C]
        labels_list: List of ground-truth labels, shape: [B, C]
        threshold: float, threshold for binarizing sigmoid probabilities
        verbose: whether to print detailed metric report

    Returns:
        metrics: dict containing f1, precision, recall, accuracy, etc.
    """
    # Concatenate all batches
    all_preds = torch.cat(preds_list, dim=0) if isinstance(preds_list, list) else preds_list
    all_labels = torch.cat(labels_list, dim=0) if isinstance(labels_list, list) else labels_list

    # Sigmoid + Binarization
    all_probs = torch.sigmoid(all_preds)
    bin_preds = (all_probs > threshold).float()

    # Convert to numpy
    y_pred = bin_preds.cpu().numpy()
    y_true = all_labels.cpu().numpy()

    # Calculate Metrics
    metrics = {
        'F1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'F1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Hamming_accuracy': 1.0 - np.mean(np.not_equal(y_true, y_pred)),
        'Exact_match_ratio': np.mean(np.all(y_true == y_pred, axis=1)),
        'Sample_wise_accuracy': (y_true == y_pred).mean(axis=1).mean()
    }

    if verbose:
        print("\n📊 Evaluation Metrics Summary:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return metrics




def plot_convergence_curves(train_losses_steps, save_path, max_points_to_plot=2000):
    """
    Plots and saves smoothed training loss convergence curves, handling large datasets via downsampling.

    Args:
        train_losses_steps (list): List of training losses per step.
        save_path (str): Path to save the plot.
        max_points_to_plot (int): Maximum points to plot for downsampling.
    """
    print(f"📈 Plotting training convergence curve ({len(train_losses_steps)} steps) and saving to {save_path}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    num_total_steps = len(train_losses_steps)
    if num_total_steps == 0:
        print("⚠️ Warning: No training loss data to plot.")
        plt.close(fig)
        return

    # --- 1. Smart Downsampling ---
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

    # --- 2. Plotting ---
    ax.plot(steps_x, losses_to_plot, color='#1f77b4', label=f'Training Loss{label_suffix}', linewidth=2)

    # --- 3. Styling ---
    ax.set_title('Training Loss Convergence Curve', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if len(losses_to_plot) > 0:
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
    Computes and plots the label co-occurrence confusion matrix.
    Cell (i, j) represents the proportion of times label j was predicted when true label was i.

    Args:
        y_true (np.array): True labels, shape [N, C].
        y_pred_binary (np.array): Binarized predicted labels, shape [N, C].
        class_names (list): List of class names, length C.
        save_path (str): Path to save the plot.
        normalize (bool): Whether to normalize rows to display percentages.
    """
    print(f"📈 Plotting label co-occurrence confusion matrix and saving to {save_path}...")
    num_classes = len(class_names)

    # Initialize C x C matrix
    co_occurrence_matrix = np.zeros((num_classes, num_classes))

    # Iterate over each class i (as the true fault)
    for i in range(num_classes):
        # Find indices where true label is i
        true_i_indices = np.where(y_true[:, i] == 1)[0]

        if len(true_i_indices) == 0:
            continue

        # Count predictions for all classes j when true class is i
        pred_counts_for_true_i = y_pred_binary[true_i_indices].sum(axis=0)

        # Fill row i
        co_occurrence_matrix[i, :] = pred_counts_for_true_i

    # --- Normalization ---
    if normalize:
        row_sums = y_true.sum(axis=0)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        matrix_to_plot = normalized_matrix
        fmt = '.2f'
        title = 'Normalized Label Co-occurrence Matrix (Recall on Diagonal)'
    else:
        matrix_to_plot = co_occurrence_matrix
        fmt = 'd'
        title = 'Label Co-occurrence Count Matrix'

    # --- Plotting ---
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
    print("✅ Co-occurrence confusion matrix saved.")