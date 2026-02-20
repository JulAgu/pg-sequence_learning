from tqdm import tqdm
import torch
from utils.utilities import save_checkpoint
from utils.tensorflowLogger import EpochWriter

class Trainer(object):
    def __init__(self,
                 exp_name,
                 type_of_model,
                 model,
                 hyperparameters,
                 train_dataloader,
                 val_dataloader,
                 checkpoints_path,
                 logs_path,
                 means_std_path,
                 device,
                 overfit=False):

        # Core components
        self.exp_name = exp_name
        self.type_of_model = type_of_model
        self.model = model.to(device)
        self.hyperparameters = hyperparameters
        self.device = device
        self.writer = EpochWriter(exp_name)

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Paths
        self.checkpoints_path = checkpoints_path
        self.logs_path = logs_path

        # Training options
        self.overfit = overfit

        # Internal trackers
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self,
                    epoch,
                    num_epochs):
        
        self.model.train()
        total_epoch_loss = 0.0
        total_epoch_mae = 0.0

        for batch in tqdm(self.train_dataloader):
            total_batch_loss = 0.0
            total_batch_mae = 0.0
            outputs = []
            if self.overfit:
                static_data_cat = self.new_batch["static_data_cat"][:10].to(self.device)
                static_data_num = self.new_batch["static_data_num"][:10].to(self.device)
                if self.type_of_model == "Hybrid":
                    meteo_ts = self.new_batch["meteo_ts"][:10].to(self.device)
                target_ts = self.new_batch["target_ts"][:10].to(self.device)
            else:
                static_data_cat = batch["static_data_cat"].to(self.device)
                static_data_num = batch["static_data_num"].to(self.device)
                if self.type_of_model == "Hybrid":
                    meteo_ts = batch["meteo_ts"].to(self.device)
                target_ts = batch["target_ts"].to(self.device)

            self.optimizer.zero_grad()
            if self.type_of_model == "Hybrid":
                outputs = self.model(static_data_num, static_data_cat, meteo_ts)
            elif self.type_of_model == "MLP":
                outputs = self.model(static_data_num, static_data_cat)
            else:
                raise ValueError("Model type not recognized")

            total_batch_loss = self.criterion_mse(outputs, target_ts)
            total_batch_mae = torch.mean(torch.abs(outputs - target_ts))
            
            total_batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.hyperparameters["max_norm"])
            self.optimizer.step()
            self.scheduler.step() # I did experiments with and without this line, Task : look af the differences

            total_epoch_loss += total_batch_loss.item()
            total_epoch_mae += total_batch_mae.item()

        print(f"TRAIN : Epoch [{epoch+1}/{num_epochs}], Loss: {total_epoch_loss:.4f}")

        checkpoint = {
            "epoch": epoch+1,
            "model": self.model.state_dict(), # model_dict before
            "optimizer": self.optimizer.state_dict(),
            }
        save_checkpoint(checkpoint, filename=f"checkpoints/{self.exp_name}/checkpoint.pth")
        tf_logs = {
            "Loss": total_epoch_loss,
            "MAE": total_epoch_mae,
            "LR": self.optimizer.param_groups[0]['lr'],
        }
        self.writer.log_epoch(epoch, "Train", tf_logs)

    def eval_epoch(self,
                   epoch):
        total_epoch_loss = 0.0
        total_epoch_mae = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                total_batch_loss = 0.0
                total_batch_mae = 0.0
                outputs = []

                static_data_cat = batch["static_data_cat"].to(self.device)
                static_data_num = batch["static_data_num"].to(self.device)
                if self.type_of_model == "Hybrid":
                    meteo_ts = batch["meteo_ts"].to(self.device)
                target_ts = batch["target_ts"].to(self.device)

                if self.type_of_model == "Hybrid":
                    outputs = self.model(static_data_num, static_data_cat, meteo_ts)
                elif self.type_of_model == "MLP":
                    outputs = self.model(static_data_num, static_data_cat)
                else:
                    raise ValueError("Model type not recognized")
                total_batch_loss = self.criterion_mse(outputs, target_ts)
                total_batch_mae = torch.mean(torch.abs(outputs - target_ts))

                total_epoch_loss += total_batch_loss.item()
                total_epoch_mae += total_batch_mae.item()   
                tf_logs = {
                    "Loss": total_epoch_loss,
                    "MAE": total_epoch_mae,
                }

                self.writer.log_epoch(epoch, "Eval", tf_logs)
                
                if total_epoch_loss < self.best_val_loss:
                    self.best_val_loss = total_epoch_loss
                    checkpoint = {
                        "epoch": epoch+1,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint,
                                    filename=f"checkpoints/{self.exp_name}/best_model.pth",
                                    best_flag=True)

    def train_loop(self):
            self.criterion_mse = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.hyperparameters["learning_rate"])

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                 max_lr=self.hyperparameters["learning_rate"],
                                                                 steps_per_epoch=len(self.train_dataloader),
                                                                 epochs=self.hyperparameters["num_epochs"])
            if self.overfit:
                iterator = iter(self.train_dataloader)
                self.new_batch = next(iterator)

            for epoch in range(self.hyperparameters["num_epochs"]):
                if self.overfit:
                    print(self.new_batch["id"][:10])

                self.train_epoch(epoch, self.hyperparameters["num_epochs"])
                if not self.overfit:
                    self.eval_epoch(epoch)