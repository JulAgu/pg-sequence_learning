import os
import pickle
import pyarrow.parquet as pq
import torch
from datasets.dataOps import create_dataloaders

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import mlflow

@hydra.main(config_path='conf',
            config_name='config',
            version_base='1.3')
def main(cfg: DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)

    os.makedirs(cfg.checkpoints_dir + cfg.exp_name, exist_ok=True) 

    data = {}
    for array in ["static_data", "before_ts", "after_ts", "target_ts", "mask_target", "cat_dicos"]:
        with open(f"{cfg.raw_data_folder + array}.pkl", "rb") as f:
            data[array] = pickle.load(f)
    table = pq.read_table(cfg.raw_data_folder + cfg.info_ts_file)
    ids = table.to_pandas().index.to_list()
    list_unic_cat = [len(dico.keys()) for dico in data["cat_dicos"].values()]

    train_dataset, val_dataset, test_dataset, _ = instantiate(cfg.training.dataset_object,
                                                              ids=ids,
                                                              static_data=data["static_data"],
                                                              before_ts=data["before_ts"],
                                                              after_ts=data["after_ts"],
                                                              target_ts=data["target_ts"],
                                                              mask_target=data["mask_target"],
                                                              train_size=cfg.training.train_size,
                                                              val_size=cfg.training.val_size,
                                                              raw_data_folder=cfg.raw_data_folder,
                                                              means_and_stds_path=cfg.means_and_stds_path)
    
    train_loader, val_loader, _ = create_dataloaders(train_dataset,
                                                     val_dataset,
                                                     test_dataset,
                                                     batch_size=cfg.training.batch_size)
    
    encoder = instantiate(cfg.model.encoder,
                          list_unic_cat=list_unic_cat)
    decoder = instantiate(cfg.model.decoder)

    optimizer = instantiate(cfg.training.optimizer,
                            params = list(encoder.parameters()) + list(decoder.parameters()))

    scheduler = instantiate(cfg.training.scheduler,
                            optimizer=optimizer,
                            steps_per_epoch=len(train_loader))

    softadapt = instantiate(cfg.training.softadapt_object)

    trainer = instantiate(cfg.training.trainer,
                          exp_name=cfg.exp_name,
                          encoder=encoder,
                          decoder=decoder,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          train_dataloader=train_loader,
                          val_dataloader=val_loader,
                          softadapt_bool=cfg.training.softadapt_bool,
                          softadapt_interval=cfg.training.softadapt_epochs_to_update,
                          softadapt_object = softadapt,
                          checkpoints_path= cfg.checkpoints_dir + cfg.exp_name,
                          logs_path=cfg.logs_dir + cfg.exp_name,
                          means_std_path=cfg.means_and_stds_path,
                          device=device
                         )

    mlflow.set_experiment(cfg.exp_name)
    with mlflow.start_run():
        mlflow.set_tag("exp_name", cfg.exp_name)
        # Log config as an artifact
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config_exp.yaml")
        # Log training parameters
        for key, value in cfg.training.items():
            mlflow.log_param(key, value)
        # Log model hyperparameters
        for key, value in cfg.model.hyperparameters.items():
            mlflow.log_param(key, value)

        trainer.train_loop()

if __name__ == "__main__":
    main()