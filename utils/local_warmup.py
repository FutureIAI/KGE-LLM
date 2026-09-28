"""Training-split descriptor fitting and independent local-branch warm-up."""

import copy

import torch


def _local_input(model, batch, device):
    signal = batch["data"].float().to(device)
    return model.normalize_layers(signal, "norm").transpose(1, 2).contiguous()


def fit_local_descriptors(model, train_loader, device):
    def normalized_training_batches():
        for batch in train_loader:
            yield _local_input(model, batch, device)

    model.moe_ffn.fit_standardization(normalized_training_batches())


def warm_up_local(model, train_loader, val_loader, loss_fn, device, epochs, learning_rate, checkpoint_path):
    """Select a local state on seen validation, then leave the joint optimizer fresh."""
    optimizer = torch.optim.Adam(model.moe_ffn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-8
    )
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.moe_ffn.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            x = _local_input(model, batch, device)
            labels = batch["label"].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model.moe_ffn(x)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(labels)
            train_count += len(labels)
        scheduler.step()

        model.moe_ffn.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x = _local_input(model, batch, device)
                labels = batch["label"].float().to(device)
                _, logits = model.moe_ffn(x)
                val_total += float(loss_fn(logits, labels)) * len(labels)
                val_count += len(labels)
        if train_count == 0 or val_count == 0:
            raise ValueError("Local warm-up requires nonempty train and validation splits")
        val_loss = val_total / val_count
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_total / train_count,
            "val_loss": val_loss,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy({
                name: value.detach().cpu() for name, value in model.moe_ffn.state_dict().items()
            })
            best_epoch = epoch + 1

    model.moe_ffn.load_state_dict(best_state, strict=True)
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "history": history,
    }, checkpoint_path)
    return {"best_epoch": best_epoch, "best_val_loss": best_loss, "history": history}
