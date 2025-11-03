import os
import pickle
import pyarrow.parquet as pq
import torch
from models.SimpleMLP import SimpleMLP
from datasets.dataOps import create_datasets, create_dataloaders
from engine.SimpleTrainer import Trainer

EXPE_NAME = "BL1_yield_MLP_100_eps"

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)

    os.makedirs(f"checkpoints/{EXPE_NAME}", exist_ok=True)

    data = {}
    for array in ["static_data", "before_ts", "after_ts", "target_ts", "mask_target", "cat_dicos"]:
        with open(f"data/{array}.pkl", "rb") as f:
            data[array] = pickle.load(f)
    table = pq.read_table("data/info_ts.parquet")
    ids = table.to_pandas().index.to_list()

    path_to_means_stds = "data/work_data/means_and_stds_mlp.pkl" # I know, it's not very elegant, but it avoids a major refactoring.

    hyperparameters = {
    # Model parameters
    "input_dim": 1619,
    "list_unic_cat": [len(dico.keys()) for dico in data["cat_dicos"].values()],
    "embedding_dims": [254, 254, 254, 254],
    "hidden_dim": 512,
    "output_dim": 1,

    # Training cycle parameters
    "batch_size": 64,
    "max_norm": 1.0,
    "learning_rate": 1e-5,
    "num_epochs": 100,
    }

    train_dataset, val_dataset, test_dataset = create_datasets(ids=ids,
                                                               static_data=data["static_data"],
                                                               before_ts=data["before_ts"],
                                                               after_ts=data["after_ts"],
                                                               target_ts=data["target_ts"],
                                                               mask_target=data["mask_target"],
                                                               train_size=0.6,
                                                               val_size=0.2,
                                                               raw_data_folder="data/",
                                                               means_and_stds_path=path_to_means_stds,
                                                               target_mode="yield",
                                                               type_of_dataset="AgrialStaticDataset"
                                                               )
    
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset,
                                                               val_dataset,
                                                               test_dataset,
                                                               batch_size=hyperparameters["batch_size"])

    model = SimpleMLP(
        input_dim=hyperparameters["input_dim"],
        list_unic_cat=hyperparameters["list_unic_cat"],
        embedding_dims=hyperparameters["embedding_dims"],
        hidden_dim=hyperparameters["hidden_dim"],
        output_dim=hyperparameters["output_dim"]
    )

    trainer = Trainer(
        exp_name=EXPE_NAME,
        type_of_model="MLP",
        model=model,
        hyperparameters=hyperparameters,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoints_path=f"checkpoints/{EXPE_NAME}",
        logs_path=f"logs/{EXPE_NAME}",
        means_std_path=path_to_means_stds,
        device=device,
    )

    trainer.train_loop()
