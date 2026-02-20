import os
import pickle
import numpy as np
import pyarrow.parquet as pq
import torch
from datasets.dataOps import create_dataloaders
from utils.utilities import load_checkpoint
from torch.utils.data import DataLoader, Subset

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import mlflow

@hydra.main(config_path='conf',
            config_name='config',
            version_base='1.3')
def main(cfg: DictConfig) -> None:
    RUN_ID = cfg.base_run_id
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.client.MlflowClient()
    dico = client.get_run(RUN_ID).to_dictionary()
    print(dico["data"]["tags"]["exp_name"])
    base_cfg = OmegaConf.load(os.path.join(dico["info"]["artifact_uri"].removeprefix("file://"), "config_exp.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)

    os.makedirs(cfg.checkpoints_dir + cfg.exp_name, exist_ok=True) 

    data = {}
    for array in ["static_data", "before_ts", "after_ts", "target_ts", "mask_target", "cat_dicos"]:
        with open(f"{base_cfg.raw_data_folder + array}.pkl", "rb") as f:
            data[array] = pickle.load(f)
    table = pq.read_table(base_cfg.raw_data_folder + base_cfg.info_ts_file)
    ids = table.to_pandas().index.to_list()
    list_unic_cat = [len(dico.keys()) for dico in data["cat_dicos"].values()]

    train_dataset, val_dataset, test_dataset, ood_dataset = instantiate(cfg.training.dataset_object,
                                                                        ids=ids,
                                                                        static_data=data["static_data"],
                                                                        before_ts=data["before_ts"],
                                                                        after_ts=data["after_ts"],
                                                                        target_ts=data["target_ts"],
                                                                        mask_target=data["mask_target"],
                                                                        train_size=base_cfg.training.train_size,
                                                                        val_size=base_cfg.training.val_size,
                                                                        raw_data_folder=base_cfg.raw_data_folder,
                                                                        means_and_stds_path=base_cfg.means_and_stds_path)
    
    # 10 random idx from ood_dataset
    selected_ids = list(np.random.randint(0, len(ood_dataset), 10))
    subset_dataset = Subset(ood_dataset, selected_ids)
    loader = DataLoader(subset_dataset,
                        batch_size=cfg.training.batch_size)


    _, val_loader, _ = create_dataloaders(train_dataset,
                                          ood_dataset,
                                          test_dataset,
                                          batch_size=cfg.training.batch_size)
    
    encoder = instantiate(base_cfg.model.encoder,
                          list_unic_cat=list_unic_cat)
    decoder = instantiate(base_cfg.model.decoder)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optimizer = instantiate(base_cfg.training.optimizer,
                            params = list(encoder.parameters()) + list(decoder.parameters()))

    scheduler = instantiate(base_cfg.training.scheduler,
                            optimizer=optimizer,
                            steps_per_epoch=len(loader))
    
    softadapt = instantiate(base_cfg.training.softadapt_object)

    # Charge the pretrained weights and biases:

    checkpoint = torch.load(dico["info"]["artifact_uri"].removeprefix("file://") + "/best_model/best_model.pth")

    load_checkpoint(checkpoint,
                    encoder,
                    decoder,
                    optimizer)


    trainer = instantiate(base_cfg.training.trainer,
                          exp_name=cfg.exp_name,
                          encoder=encoder,
                          decoder=decoder,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          train_dataloader=loader,
                          val_dataloader=val_loader,
                          num_epochs = cfg.training.num_epochs,
                          softadapt_bool=base_cfg.training.softadapt_bool,
                          softadapt_interval=base_cfg.training.softadapt_epochs_to_update,
                          softadapt_object = softadapt,
                          checkpoints_path= cfg.checkpoints_dir + cfg.exp_name,
                          logs_path=cfg.logs_dir + cfg.exp_name,
                          means_std_path=base_cfg.means_and_stds_path,
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