import argparse
import logging
import os

import lightning as L
import lightning.pytorch as Lpt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification

from dataset import get_dataset_for_training
from lmodule import ClassificationModule, SoftmaxClassificationModule
from model import SimpleClassifierModel, SimpleClassifierModelWithBN, SimpleClassifierModelWithBNSELU

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_model(model_str, pretrained_model, freeze):
    if model_str == "MLP":
        return SimpleClassifierModel(pretrained_model, freeze=freeze)
    elif model_str == "MLPSELU":
        return SimpleClassifierModelWithBNSELU(pretrained_model, freeze=freeze)
    elif model_str == "MLPBN":
        return SimpleClassifierModelWithBN(pretrained_model, freeze=freeze)
    elif model_str == "Auto":
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)
        if freeze:
            for param in model.base_model.parameters():
                param.requires_grad = False
        return model
    else:
        raise Exception("Unknown model")


def get_module(model, loss_str, lr):
    if loss_str == "Logistic":
        return ClassificationModule(model, lr=lr)
    elif loss_str.startswith("Focal"):
        my_alpha = float(loss_str[5:])
        from torchvision.ops import sigmoid_focal_loss
        return ClassificationModule(model, lr=lr,
                                    loss=lambda a, b: sigmoid_focal_loss(a, b, alpha=my_alpha, reduction="mean"))

    elif loss_str == "Softmax":
        return SoftmaxClassificationModule(model, lr=lr)
    else:
        raise Exception("Unknown loss")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--loss", type=str, default="Logistic")
    parser.add_argument("--model", type=str, default="MLP")
    parser.add_argument("--freeze", action='store_true', help="Freeze the pretrained model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrained_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)

    args = parser.parse_args()

    Lpt.seed_everything(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{args.outdir}/run.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Preparing training data... ")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    tokenized_datasets = get_dataset_for_training(tokenizer, path_prefix=args.outdir)

    # Data Loaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator, num_workers=8
    )
    val_dataloader = DataLoader(
        tokenized_datasets["val"], batch_size=args.batch_size * 4, collate_fn=data_collator, num_workers=8
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=args.batch_size * 4, collate_fn=data_collator, num_workers=8
    )
    # Setup training
    logging.info("Setup training... ")
    model = get_model(args.model, args.pretrained_model, args.freeze)

    l_module = get_module(model, args.loss, args.lr)

    wandb_logger = Lpt.loggers.WandbLogger(project="critics-classifier")

    wandb_logger.watch(model)

    trainer = L.Trainer(max_steps=-1, devices=-1, strategy='ddp_find_unused_parameters_true',
                        limit_train_batches=args.limit_train_batches,
                        # log_every_n_steps=500,
                        val_check_interval=1000, logger=wandb_logger,  # precision=16,
                        callbacks=[
                            Lpt.callbacks.ModelCheckpoint(dirpath=os.path.join(args.outdir, "my_cpts/"), save_top_k=2,
                                                          monitor="val_loss"),
                            Lpt.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min",
                                                                       patience=500)]
                        )

    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(args)
    # Train
    trainer.fit(l_module, train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.ckpt_path)

    trainer.test(l_module, dataloaders=test_dataloader)
