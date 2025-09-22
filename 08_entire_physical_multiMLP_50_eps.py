import os
import pickle
import pyarrow.parquet as pq
import torch
from models.EncoderDecoderIndependentMLP import Encoder, Decoder
from datasets.dataOps import create_datasets, create_dataloaders
from engine.EntireTrainer import Trainer

EXPE_NAME = "13_entire_physical_multiMLP_50_eps_NEW_HYPERPARAMS"

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)

    os.makedirs(f"checkpoints/{EXPE_NAME}", exist_ok=True)

    data = {}
    for array in ["static_data", "before_ts", "after_ts", "target_ts", "mask_target", "cat_dicos"]:
        with open(f"data/big_agrial_ts_wise/{array}.pkl", "rb") as f:
            data[array] = pickle.load(f)
    table = pq.read_table("data/info_ts.parquet")
    ids = table.to_pandas().index.to_list()

    # hyperparameters = {
    # # Static encoder parameters
    # "static_input_dim": 9,
    # "list_unic_cat": [len(dico.keys()) for dico in data["cat_dicos"].values()],
    # "embedding_dims": [100, 100, 100, 100],
    # "hidden_dim_static_encoder": 128,

    # # Dynamic encoder parameters
    # "dynamic_input_dim": 7,
    # "hidden_dim_dynamic_encoder": 256,
    # "first_decoder_input_dim": 11,
    # "gru_encoder_num_layers": 2,

    # # Decoder parameters
    # "gru_input_dim": 11,
    # "gru_hidden_dim": 256 + 128,  # hidden_dim_dynamic_encoder + hidden_dim_static_encoder
    # "stepwise_input_dim": 7,
    # "main_hidden_dim": 128,
    # "mask_hidden_dim": 128,
    # "output_dim": 11,
    # "gru_decoder_num_layers": 2,

    # # Training cycle parameters
    # "batch_size": 64,
    # "teacher_forcing_ratio": 0.7,
    # "max_norm": 1.0,
    # "learning_rate": 1e-4,
    # "num_epochs": 50,
    # "alpha": 0.7,
    # "beta": 0.15,
    # "gamma": 0.15,
    # }

    hyperparameters = {
    # Static encoder parameters
    "static_input_dim": 9,
    "list_unic_cat": [len(dico.keys()) for dico in data["cat_dicos"].values()],
    "embedding_dims": [150, 150, 150, 150],
    "hidden_dim_static_encoder": 256,

    # Dynamic encoder parameters
    "dynamic_input_dim": 7,
    "hidden_dim_dynamic_encoder": 384,
    "first_decoder_input_dim": 11,
    "gru_encoder_num_layers": 2,

    # Decoder parameters
    "gru_input_dim": 11,
    "gru_hidden_dim": 384 + 256,  # hidden_dim_dynamic_encoder + hidden_dim_static_encoder
    "stepwise_input_dim": 7,
    "main_hidden_dim": 256,
    "mask_hidden_dim": 256,
    "output_dim": 11,
    "gru_decoder_num_layers": 2,

    # Training cycle parameters
    "batch_size": 64,
    "teacher_forcing_ratio": 0.7,
    "max_norm": 1.0,
    "learning_rate": 1e-4,
    "num_epochs": 60,
    "alpha": 0.7,
    "beta": 0.15,
    "gamma": 0.15,
    }

    train_dataset, val_dataset, test_dataset = create_datasets(ids=ids,
                                                               static_data=data["static_data"],
                                                               before_ts=data["before_ts"],
                                                               after_ts=data["after_ts"],
                                                               target_ts=data["target_ts"],
                                                               mask_target=data["mask_target"],
                                                               train_size=0.6,
                                                               val_size=0.2,
                                                               raw_data_folder="data/big_agrial_ts_wise/"
                                                               )
    
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset,
                                                               val_dataset,
                                                               test_dataset,
                                                               batch_size=hyperparameters["batch_size"])

    encoder = Encoder(
        static_input_dim=hyperparameters["static_input_dim"],
        static_hidden_dim=hyperparameters["hidden_dim_static_encoder"],
        list_unic_cat=hyperparameters["list_unic_cat"],
        embedding_dims=hyperparameters["embedding_dims"],
        dynamic_input_dim=hyperparameters["dynamic_input_dim"],
        dynamic_hidden_dim=hyperparameters["hidden_dim_dynamic_encoder"],
        first_decoder_input_dim=hyperparameters["first_decoder_input_dim"],
        gru_num_layers=hyperparameters["gru_encoder_num_layers"]
    )

    decoder = Decoder(
        gru_input_dim=hyperparameters["gru_input_dim"],
        gru_hidden_dim=hyperparameters["gru_hidden_dim"],
        stepwise_input_dim=hyperparameters["stepwise_input_dim"],
        main_hidden_dim=hyperparameters["main_hidden_dim"],
        output_dim=hyperparameters["output_dim"],
        num_layers=hyperparameters["gru_decoder_num_layers"]
    )

    trainer = Trainer(
        exp_name=EXPE_NAME,
        encoder=encoder,
        decoder=decoder,
        hyperparameters=hyperparameters,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoints_path=f"checkpoints/{EXPE_NAME}",
        logs_path=f"logs/{EXPE_NAME}",
        monotonicity_bool=False,
        static_bool=True,
        dynamic_bool=True,
        means_std_path="data/big_agrial_ts_wise/means_and_stds.pkl",
        device=device
    )

    trainer.train_loop()
