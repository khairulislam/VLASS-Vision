from typing import Any, Optional
from lightning import LightningModule
from lightning.pytorch.cli import (
    ArgsType,
    LightningCLI,
    LRSchedulerTypeUnion,
)
from torch.optim import Optimizer
from lightning.pytorch.cli import SaveConfigCallback
import os, datetime, timm
from lightning.pytorch.loggers import WandbLogger
from lightning import Trainer

class CustomSaveConfigCallback(SaveConfigCallback):
    """Saves full training configuration
    Otherwise wandb won't log full configuration but only flattened module and data hyperparameters
    """

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        for logger in trainer.loggers:
            if issubclass(type(logger), WandbLogger):
                logger.experiment.config.update(self.config.as_dict())
        return super().save_config(trainer, pl_module, stage)

class WrappedLightningCLI(LightningCLI):

    # Changing the lr_scheduler interval to step instead of epoch
    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: Optimizer,
        lr_scheduler: Optional[LRSchedulerTypeUnion] = None,
    ) -> Any:
        optimizer_list, lr_scheduler_list = LightningCLI.configure_optimizers(
            lightning_module, optimizer=optimizer, lr_scheduler=lr_scheduler
        )

        for idx in range(len(lr_scheduler_list)):
            if not isinstance(lr_scheduler_list[idx], dict):
                lr_scheduler_list[idx] = {
                    "scheduler": lr_scheduler_list[idx],
                    "interval": "step",
                }
        return optimizer_list, lr_scheduler_list


def main_cli(args: ArgsType = None, run: bool = True):
    # returns SLURM_NTASKS not found error for the multimodal training
    print(f"Training started at {datetime.datetime.now()}")
    
    cli = WrappedLightningCLI(
        save_config_kwargs={"overwrite": True},
        args=args,
        run=run
    )
    return cli


if __name__ == "__main__":
    main_cli(run=True)
