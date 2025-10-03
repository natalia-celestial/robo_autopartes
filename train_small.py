# train_small.py
from rfdetr import RFDETRBase
m = RFDETRBase()
m.train(
    dataset_dir="dataset/images",
    output_dir="runs/rfdetr_autoparts_small",
    resolution=560,
    batch_size=1,
    grad_accum_steps=8,
    multi_scale=False,
    expanded_scales=False,
    freeze_encoder=True,
    gradient_checkpointing=True,
    num_queries=150,
    run_test=False,
    epochs=80,
    lr=1e-4,
    device="cuda",
    early_stopping=True
)
