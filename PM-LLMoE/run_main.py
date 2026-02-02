import argparse
import time
import random
import numpy as np
import os
import logging
import torch
from torch import optim
from torch.optim import lr_scheduler
from accelerate import Accelerator
from tqdm import tqdm

from models import Autoformer, DLinear, PM_LLMoE
from data_provider.data_factory import data_provider
from utils.tools import (
    del_files, EarlyStopping, adjust_learning_rate, vali_classify,
    load_content, test_classify, evaluate_metrics,
    plot_co_occurrence_matrix, plot_convergence_curves
)
from utils.losses import DynamicFocalLoss

# Initialize Logging
log = logging.getLogger()
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# Environment Settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description='PM-LLMoE Training Script')

# Basic Config
parser.add_argument('--task_name', type=str, required=True, default='classification',
                    help='Task name options: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='Status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='Model ID')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='Prefix for saving results')
parser.add_argument('--model', type=str, required=True, default='PM_LLMoE',
                    help='Model name options: [Autoformer, DLinear, PM_LLMoE]')
parser.add_argument('--seed', type=int, default=2021, help='Random seed')

# Data Loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='Dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='Root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='Data file name')
parser.add_argument('--features', type=str, default='M',
                    help='Forecasting task options: [M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='Target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='Dataset loader type')
parser.add_argument('--freq', type=str, default='h',
                    help='Freq for time features encoding options: [s, t, h, d, b, w, m]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')

# Classification Task
parser.add_argument('--num_classes', type=int, default=21, help='Number of classes')
parser.add_argument('--seq_len', type=int, default=1024, help='Input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='Label length')
parser.add_argument('--pred_len', type=int, default=0, help='Prediction sequence length (not used for classification)')
parser.add_argument('--debug', action='store_true', help='Enable debug mode to limit number of iters per epoch')
parser.add_argument('--debug_iters', type=int, default=1000, help='Maximum iters per epoch in debug mode')

# Model Definitions
parser.add_argument('--enc_in', type=int, default=21, help='Encoder input size')
parser.add_argument('--dec_in', type=int, default=21, help='Decoder input size')
parser.add_argument('--c_out', type=int, default=21, help='Output size')
parser.add_argument('--d_model', type=int, default=32, help='Dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='Dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='Window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='Attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='Time features encoding options: [timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
parser.add_argument('--output_attention', action='store_true', help='Whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='Patch length')
parser.add_argument('--stride', type=int, default=256, help='Stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='Prompt domain')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')
parser.add_argument('--content', type=str, default='TimeLLM run', help='Description for the model run')

# Optimization
parser.add_argument('--num_workers', type=int, default=10, help='Data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='Experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='Train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='Alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='Experiment description')
parser.add_argument('--loss', type=str, default='MSE', help='Loss function')
parser.add_argument('--lradj', type=str, default='type1', help='Adjust learning rate strategy')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

# MoE Settings
parser.add_argument('--moe_loss_weight', type=float, default=0.01)
parser.add_argument('--moe_alpha', type=float, default=0.01)

args = parser.parse_args()

SHORT_CLASS_NAMES = [
    "M0", "M1", "M2", "M3", "M4",
    "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
    "LA0", "LA1", "LA2", "LA3", "LA4",
    "RA0", "RA1"
]

# Set Random Seed
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
log.info(f"Random seed set to {fix_seed}")

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")
accelerator = Accelerator()

MAX_ITERS_PER_EPOCH = args.debug_iters if getattr(args, "debug", False) else float('inf')

for ii in range(args.itr):
    # Setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
        args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.des, ii)

    log.info(f"Experiment setting: {setting}")
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    os.makedirs(path, exist_ok=True)

    log_file_path = os.path.join(path, "training_log.csv")
    with open(log_file_path, "w") as f:
        f.write("epoch,train_loss,vali_loss,vali_acc\n")

    log.info("Loading datasets...")
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    log.info("Building model...")
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'ShapeFormer':
        pass  # Placeholder for ShapeFormer
    else:
        model = PM_LLMoE.Model(args).float()

    model.to(device)
    log.info(f"Model {args.model} initialized and moved to device.")

    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    args.content = load_content(args)
    if not os.path.exists(path):
        os.makedirs(path)
    log.info(f"Checkpoints path: {path}")

    time_now = time.time()
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience)

    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    # Learning Rate Scheduler
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = DynamicFocalLoss(alpha=1.0, gamma_base=1.0, gamma_scale=2.0)

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    log.info("Start training...")
    train_loss_history = []
    vali_loss_history = []

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()
        time_now = time.time()

        total_samples = 0
        total_correct = 0

        # Loss weights
        alpha = 0.8
        beta = 0.2

        for i, batch in enumerate(tqdm(train_loader, desc=f"[Train Epoch {epoch + 1}]")):
            if i >= MAX_ITERS_PER_EPOCH:
                break
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch['data'].float().to(device)
            batch_y = batch['label'].float().to(device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    # ===== forward =====
                    outputs, llm_logits, moe_logits = model(batch_x)

                    # ===== multi-head split (Dataset Specific) =====
                    motor_out = outputs[:, 0:5]
                    gear_out = outputs[:, 5:14]
                    LA_out = outputs[:, 14:19]
                    RA_out = outputs[:, 19:21]

                    motor_y = batch_y[:, 0:5]
                    gear_y = batch_y[:, 5:14]
                    LA_y = batch_y[:, 14:19]
                    RA_y = batch_y[:, 19:21]

                    # ===== multi-head loss =====
                    motor_loss = criterion(motor_out, motor_y)
                    gear_loss = criterion(gear_out, gear_y)
                    LA_loss = criterion(LA_out, LA_y)
                    RA_loss = criterion(RA_out, RA_y)

                    weights = {"motor": 1.0, "gear": 1.0, "LA": 1.0, "RA": 0.5}

                    weighted_loss = (
                                            weights["motor"] * motor_loss +
                                            weights["gear"] * gear_loss +
                                            weights["LA"] * LA_loss +
                                            weights["RA"] * RA_loss
                                    ) / (sum(weights.values()) + 1e-12)

                    # ===== LLM & MoE loss =====
                    llm_loss = criterion(llm_logits, batch_y)
                    moe_loss = criterion(moe_logits, batch_y)

                    # ===== Print Info (Sampled) =====
                    if (i + 1) % 100 == 0:
                        print(f"🔵 LLM loss = {llm_loss.item():.6f}")
                        print(f"🟢 MoE loss = {moe_loss.item():.6f}")
                        print(f"⚫ Weighted loss = {weighted_loss.item():.6f}")

                    # ===== Final loss =====
                    loss = weighted_loss + alpha * llm_loss + beta * moe_loss
                    train_loss_history.append(loss.item())

                train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.unscale_(model_optim)
                scaler.step(model_optim)
                scaler.update()

            else:
                # Fallback or standard FP32 training logic can be placed here if needed
                pass

            # === Calculate Metrics ===
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct = (preds == batch_y).float().sum()
            total_correct += correct.item()
            total_samples += batch_y.numel()

            if args.lradj == 'TST':
                adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

            if (i + 1) % 100 == 0:
                acc = total_correct / total_samples

                print(f"\n[Batch {i + 1} @ Epoch {epoch + 1}]")
                print(f"  ┣━ Total loss: {loss.item():.6f} | Accuracy: {acc:.4f}")
                print(f"  ┣━ Head losses:")
                print(f"  ┃   Motor: {motor_loss.item():.6f}")
                print(f"  ┃   Gear : {gear_loss.item():.6f}")
                print(f"  ┃   LA   : {LA_loss.item():.6f}")
                print(f"  ┃   RA   : {RA_loss.item():.6f}")

                total_head_loss = motor_loss + gear_loss + LA_loss + RA_loss
                eps = 1e-8
                print(f"  ┣━ Contribution ratio:")
                print(f"  ┃   Motor: {motor_loss.item() / (total_head_loss.item() + eps):.3f}")
                print(f"  ┃   Gear : {gear_loss.item() / (total_head_loss.item() + eps):.3f}")
                print(f"  ┃   LA   : {LA_loss.item() / (total_head_loss.item() + eps):.3f}")
                print(f"  ┃   RA   : {RA_loss.item() / (total_head_loss.item() + eps):.3f}")

                # Print MoE activation probabilities
                if hasattr(model, "last_moe_scores") and model.last_moe_scores is not None:
                    with torch.no_grad():
                        probs = model.last_moe_scores.mean(dim=(0, 1))
                        probs_cpu = probs.cpu().numpy().tolist()
                        print(f"  ┣━ MoE mean probs: {['%.3f' % p for p in probs_cpu]}")

        print(f"🕓 Epoch {epoch + 1} finished in {time.time() - epoch_time:.2f} seconds")
        train_loss = np.average(train_loss)

        # === Validation & Testing ===
        vali_loss, vali_acc = vali_classify(
            args=args,
            model=model,
            vali_data=vali_data,
            vali_loader=vali_loader,
            criterion=criterion,
            accelerator=accelerator
        )

        vali_loss_history.append(vali_loss)

        print(
            f"📉 Epoch {epoch + 1} | Train Loss: {train_loss:.6f} | Vali Loss: {vali_loss:.6f} | Vali Acc: {vali_acc:.6f}")

        with open(log_file_path, "a") as f:
            f.write(f"{epoch + 1},{train_loss},{vali_loss},{vali_acc}\n")

        # === Save Checkpoints ===
        epoch_ckpt = os.path.join(path, f"epoch_{epoch + 1}.pt")
        if accelerator is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), epoch_ckpt)
        else:
            torch.save(model.state_dict(), epoch_ckpt)

        print(f"💾 Saved checkpoint: {epoch_ckpt}")
        early_stopping(vali_loss, model, path)

        if early_stopping.early_stop:
            print("⛔ Early stopping triggered")
            # Plot convergence curves
            plot_convergence_curves(
                train_losses_steps=train_loss_history,
                save_path=os.path.join(path, "training_convergence_curve.png")
            )

            # Load best model for final evaluation
            print("🎯 Loading best model from checkpoint for final test evaluation...")
            checkpoint_path = os.path.join(path, 'checkpoint.pt')

            if accelerator is not None:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.load_state_dict(torch.load(checkpoint_path))
            else:
                model.load_state_dict(torch.load(checkpoint_path))

            model.eval()
            test_loss, test_acc, all_preds, all_labels = test_classify(
                args=args,
                accelerator=accelerator,
                model=model,
                test_loader=test_loader,
                criterion=criterion
            )

            print(f"🧪 Final Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.6f}")
            evaluate_metrics(
                preds_list=[torch.tensor(all_preds)],
                labels_list=[torch.tensor(all_labels)],
                threshold=0.5,
                verbose=True
            )
            preds_binary = (torch.sigmoid(torch.tensor(all_preds)) > 0.5).int().numpy()

            # Plot smoothed convergence
            plot_convergence_curves(
                train_losses_steps=train_loss_history,
                save_path=os.path.join(path, "convergence_curves_smooth.png")
            )
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    if not early_stopping.early_stop:
        print("⚠️ EarlyStopping not triggered, loading final model for evaluation...")

        checkpoint_path = os.path.join(path, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            if accelerator is not None:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.load_state_dict(torch.load(checkpoint_path))
            else:
                model.load_state_dict(torch.load(checkpoint_path))

            plot_convergence_curves(
                train_losses_steps=train_loss_history,
                save_path=os.path.join(path, "training_convergence_curve.png")
            )

            model.eval()
            test_loss, test_acc, all_preds, all_labels = test_classify(
                args=args,
                accelerator=accelerator,
                model=model,
                test_loader=test_loader,
                criterion=criterion
            )

            print(f"🧪 Final Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.6f}")
            evaluate_metrics(
                preds_list=[torch.tensor(all_preds)],
                labels_list=[torch.tensor(all_labels)],
                threshold=0.5,
                verbose=True
            )
            preds_binary = (torch.sigmoid(torch.tensor(all_preds)) > 0.5).int().numpy()

            plot_co_occurrence_matrix(
                y_true=all_labels,
                y_pred_binary=preds_binary,
                class_names=SHORT_CLASS_NAMES,
                save_path=os.path.join(path, "co_occurrence_matrix.png"),
                normalize=True
            )
        else:
            print("🚫 Model checkpoint not found, skipping evaluation.")

# Clear Cache
torch.cuda.empty_cache()