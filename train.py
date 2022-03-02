import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data, MyData
from module import CIFAR10Module, MyModule


def main(args):
    # if bool(args.download_weights):
    #     CIFAR10Data.download_weights()
    # else:
    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.save_path = f'result/{args.classifier}_{args.exp_name}'
    if args.logger == "wandb":
        logger = WandbLogger(name=args.classifier, project=f"{args.save_path}")
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger(f"{args.save_path}", name=args.classifier)

    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False, dirpath=args.save_path)

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger if not bool(args.dev + args.test_phase) else None,
        gpus=args.gpu_id,  # -1
        deterministic=True,
        accelerator='ddp',
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        checkpoint_callback=True,
        callbacks=[checkpoint],
        weights_save_path=args.save_path,
        precision=args.precision,
    )


    # data = CIFAR10Data(args)
    # model = CIFAR10Module(args)
    data = MyData(args)
    num_classes = len(data.classes)
    model = MyModule(args, num_classes)

    if bool(args.pretrained):
        state_dict = os.path.join(
            "state_dicts", args.classifier + ".pt"
        )
        model.model.load_state_dict(torch.load(state_dict))

    if bool(args.test_phase):
        trainer.test(model, data.test_dataloader())
    else:
        trainer.fit(model, data)
        trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--train", type=str, default="/workdir/qiuyurui/data/traffic_light/new_crop/train.txt")
    parser.add_argument("--test", type=str, default="/workdir/qiuyurui/data/traffic_light/new_crop/test.txt")
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--exp_name", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--gpu_id", type=str, default="0,1")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()
    main(args)
