from typing import Literal

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pl_models.pl_model_prototype import LitModel_with_dataloader
from utils import arguments


def train(model: LitModel_with_dataloader, opt: arguments.Train_Option, mode: Literal["train", "eval"] = "train", full_val=False):
    if not opt.debug:
        try:
            model = torch.compile(model)  # type: ignore
        except Exception:
            print("Could not compile, running normally")
    monitor_str = opt.monitor
    print("Monitoring:", monitor_str)

    checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}_latest",
        monitor=monitor_str,
        mode="min",
        save_last=True,
        save_top_k=3,
        auto_insert_metric_name=True,
        every_n_train_steps=opt.save_every_samples // opt.batch_size_effective,
    )

    # early_stopping = EarlyStopping(
    #    monitor=monitor_str,
    #    mode="min",
    #    verbose=False,
    #    patience=opt.early_stopping_patience,
    #    # check_on_train_epoch_end=True,
    # )
    resume = None
    if not opt.new:
        checkpoint_path = arguments.get_latest_checkpoint(opt, "*", opt.log_dir)
        if checkpoint_path is not None:
            resume = checkpoint_path
            print(f"Resuming from {resume}")
    logger = TensorBoardLogger(opt.log_dir, name=opt.experiment_name, default_hp_metric=False)

    n_overfit_batches = 1 if opt.overfit else 0.0

    log_every_n_steps = 1 if opt.overfit else opt.log_every_n_steps // opt.batch_size_effective
    gpus = opt.gpus
    accelerator = "gpu"
    if gpus is None:
        gpus = 1
        nodes = 1
    elif -1 in gpus:
        gpus = None
        nodes = 1
        accelerator = "cpu"
    else:
        nodes = len(gpus)
    batches = len(model.prepare_data())
    if log_every_n_steps >= batches:
        log_every_n_steps = None

    trainer = Trainer(
        max_steps=opt.total_samples // opt.batch_size_effective,
        devices=gpus,  # type: ignore
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if not opt.fp32 else 32,
        callbacks=[checkpoint],
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        overfit_batches=n_overfit_batches,
        fast_dev_run=opt.fast_dev_run,
        val_check_interval=min(opt.val_check_interval / batches, 1.0) if not full_val else None,
        limit_val_batches=opt.limit_val_batches,
    )

    if mode == "train":
        trainer.fit(model, ckpt_path=resume)
    elif mode == "eval":
        raise NotImplementedError(mode)
        ## load the latest checkpoint
        ## perform lpips
        ## dummy loader to allow calling 'test_step'
        # dummy = DataLoader(TensorDataset(torch.tensor([0.0] * opt.batch_size)), batch_size=opt.batch_size)
        # eval_path = opt.eval_path or checkpoint_path
        ## conf.eval_num_images = 50
        # print("loading from:", eval_path)
        # state = torch.load(eval_path, map_location="cpu")
        # print("step:", state["global_step"])
        # model.load_state_dict(state["state_dict"])
        ## trainer.fit(model)
        # out = trainer.test(model, dataloaders=dummy)
        # if len(out) == 0:
        #    # no results where returned
        #    return
        ## first (and only) loader
        # out = out[0]
        # print(out)
        #
        # if get_rank() == 0:
        #    # save to tensorboard
        #    for k, v in out.items():
        #        model.log(k, v)
        #
        #    # # save to file
        #    # # make it a dict of list
        #    # for k, v in out.items():
        #    #     out[k] = [v]
        #    tgt = f"evals/{opt.name}.txt"
        #    dirname = os.path.dirname(tgt)
        #    if not os.path.exists(dirname):
        #        os.makedirs(dirname)
        #    with open(tgt, "a") as f:
        #        f.write(json.dumps(out) + "\n")
        #    # pd.DataFrame(out).to_csv(tgt)
    else:
        raise NotImplementedError()
