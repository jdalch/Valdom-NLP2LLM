# Split the book into training and validation data 
train_ratio = 0.90
split_idx = int(train_ratio * len(book))
train_data = book[:split_idx]
val_data = book[split_idx:]
print(len(train_data), len(val_data))


torch.manual_seed(44)

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)