import logging
from data_preparation import VisualMRCPrepper
from datamodule import MultiModalRetrieverDataModule
import torch
from parser_utils import get_args
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint
)
from pytorch_lightning.loggers import TensorBoardLogger
import configargparse
from copy import deepcopy
from models import MultiModalRetriever
from pathlib import Path

# for memory optimization
# from lightning.pytorch.accelerators import find_usable_cuda_devices
# torch.set_float32_matmul_precision(precision="medium")
# torch.cuda.empty_cache()

# init logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:\t%(message)s"
)

# Ignore future warnings (e.g., device cuda will be deprecated)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

if __name__ == "__main__":

    # Collect main args
    parser = get_args()

    # Add arguments of other modules
    parser = VisualMRCPrepper.add_argparse_args(parent_parser=parser)
    parser = MultiModalRetrieverDataModule.add_dataloader_specific_args(parent_parser=parser)
    parser = MultiModalRetriever.add_argparse_args(parent_parser=parser)

    # Parse known args and print out unknown argss
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Unknown args: {unknown}")

    # Devices
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # device = "cuda" if args.cuda else "cpu"
    import os
    # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # device = "mps"
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    device = "cpu"

    # Workaround to parse args from config file twice
    _actions = deepcopy(parser._actions)
    _option_string_actions = deepcopy(parser._option_string_actions)
    parser = configargparse.ArgParser(default_config_files=[args.my_config])
    parser._actions = _actions
    parser._option_string_actions = _option_string_actions
    # End workaround

    # If user loads original splits, then format them (only needs to be done once)
    if not args.load_multimodal_csv_data:
        logger.info('Note that you provided the data in the format of VisualMRC so first it will be formatted in the '
                    'required format for the MultiModalRetriever')
        dataset_prepper = VisualMRCPrepper.from_argparse_args(args=args)

        # Prepare the data in the MultiModalDPR format and save it to CSV
        for split_name in ["train", "val", "test"]:
            logger.info(f'Parsing the VisualMRC {split_name} data')
            prepped_data = dataset_prepper.parse_visualmrc_data(
                data_folder_path=args.data_folder,
                split_name=split_name
            )
            dataset_prepper.save_to_csv(
                data=prepped_data,
                data_folder_path=args.data_folder,
                split_name=split_name
            )
            
    # Otherwise use the already formatted CSVs
    else:
        logger.info(f'The option to load pre-formatted data was chosen, skipping the preparation step')

    # Setting up DataModule
    datamodule = MultiModalRetrieverDataModule.from_argparse_args(
        args=args,
        data_folder_path=args.data_folder
    )
    
    # set datamodule for train and validation
    datamodule.setup(stage="fit")
    num_batches_per_epoch = len(
        datamodule.train_dataset.to_dataloader(
            batch_size=args.batch_size, shuffle=True, drop_last=True
        )
    )

    # Instantiate MultiModalRetriever
    model = MultiModalRetriever.from_argparse_args(args=args, num_batches_per_epoch=num_batches_per_epoch)
    
    # CALLBACKS
    # early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss_epoch",
        verbose=True
    )
    
    # checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        verbose=True,
        dirpath=Path("experiments", "best_models", args.modality, f"bs={args.batch_size}-nnr={args.num_neg_rois}-ts={args.num_tot_samples}")
    )
    
    # initiate loggers
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        save_dir="experiments",
        name=args.modality,
        version=fr"bs={args.batch_size}-nnr={args.num_neg_rois}-agb={args.acc_grad_batches}"
    )

    # bundle up callbacks
    callbacks = [
        lr_logger,
        early_stopping,
        checkpoint_callback
    ]

    # Instantiate PL Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=device,
        devices=1,
        val_check_interval=0.4,
        accumulate_grad_batches=args.acc_grad_batches,
        precision=args.precision,
        callbacks=callbacks,
        gradient_clip_val=args.grad_clip_val,
        logger=tb_logger,
        enable_model_summary=True,
        enable_progress_bar=True
    )
    
    # TRAINING
    # fit trainer
    logger.info(f"Training with {args.modality} modality")
    logger.info("Start training ...")
    # torch.cuda.empty_cache()
    trainer.fit(
        model=model,
        datamodule=datamodule
    )
    
    # TESTING
    # set datamodule for test
    datamodule.setup(stage="test")
    
    # test trainer
    logger.info("Testing version with best weights on test set ...")
    trainer.test(
        model=model,
        datamodule=datamodule
    )
