from tqdm import tqdm
import torch
import pickle
from utils.schedulers import cosine_scheduler, teacher_forcing_decay
from utils.losses import monotone_penalty, biomass_penalty, assimilation_penalty, dry_matter_increase_penalty
from utils.utilities import save_checkpoint, load_checkpoint
from utils.tensorflowLogger import EpochWriter

class Trainer(object):
    def __init__(self,
                 exp_name,
                 encoder,
                 decoder,
                 hyperparameters,
                 train_dataloader,
                 val_dataloader,
                 checkpoints_path,
                 logs_path,
                 monotonicity_bool,
                 static_bool,
                 dynamic_bool,
                 means_std_path,
                 device,
                 overfit=False):

        # Core components
        self.exp_name = exp_name
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.hyperparameters = hyperparameters
        self.device = device
        self.writer = EpochWriter(exp_name)

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        with open(means_std_path, "rb") as f:
            self.means_and_stds_dict = pickle.load(f)
        self.target_ts_mean = torch.from_numpy(self.means_and_stds_dict["target_ts_mean"]).to(device)
        self.target_ts_std = torch.from_numpy(self.means_and_stds_dict["target_ts_std"]).to(device)

        # Paths
        self.checkpoints_path = checkpoints_path
        self.logs_path = logs_path

        # Training options
        self.monotonicity_bool = monotonicity_bool
        self.static_bool = static_bool
        self.dynamic_bool = dynamic_bool
        self.overfit = overfit

        # Internal trackers
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self,
                    epoch,
                    num_epochs):
        
        self.encoder.train()
        self.decoder.train()

        if self.monotonicity_bool:
            epoch_MonoParam = cosine_scheduler(epoch,
                                          num_epochs,
                                          beta_max=self.hyperparameters["MonoParam"])
        tf_ratio = teacher_forcing_decay(epoch, num_epochs, end_ratio=self.hyperparameters["teacher_forcing_ratio"])
        total_epoch_loss = 0.0
        total_epoch_data_loss = 0.0
        total_epoch_mono_loss = 0.0
        total_epoch_biomass_loss = 0.0
        total_epoch_assimilation_loss = 0.0
        total_epoch_dry_matter_increase_loss = 0.0

        for batch in tqdm(self.train_dataloader):
            total_batch_data_loss = 0.0
            total_batch_loss = 0.0
            mono_loss = 0.0
            biomass_loss = 0.0
            assimilation_loss = 0.0
            dry_matter_increase_loss = 0.0
            outputs = []
            if self.overfit:
                static_data_cat = self.new_batch["static_data_cat"][:3].to(self.device)
                static_data_num = self.new_batch["static_data_num"][:3].to(self.device)
                before_ts = self.new_batch["before_ts"][:3].to(self.device)
                after_ts = self.new_batch["after_ts"][:3].to(self.device)
                target_ts = self.new_batch["target_ts"][:3].to(self.device)
                mask_target = self.new_batch["mask_target"][:3].to(self.device)
            else:
                static_data_cat = batch["static_data_cat"].to(self.device)
                static_data_num = batch["static_data_num"].to(self.device)
                before_ts = batch["before_ts"].to(self.device)
                after_ts = batch["after_ts"].to(self.device)
                target_ts = batch["target_ts"].to(self.device)
                mask_target = batch["mask_target"].to(self.device)

            self.optimizer.zero_grad()
            latent, x_t = self.encoder(static_data_num, static_data_cat, before_ts)
            h_t = latent  # h_0
            for t in range(after_ts.shape[1]):
                output, h_t, _ = self.decoder(x_t.unsqueeze(1), h_t, after_ts[:, t, :])
                outputs.append(output.unsqueeze(1))

                if torch.rand(1).item() < tf_ratio:
                    x_t = target_ts[:, t, :]
                else:
                    x_t = output

                mask_t = mask_target[:, t]
                loss_unreduced = self.criterion_mse(output, target_ts[:, t, :])
                loss_masked = loss_unreduced * mask_t
                data_loss = loss_masked.sum() / (mask_t.sum() + 1e-8)
                total_batch_data_loss += data_loss
                if self.monotonicity_bool or self.static_bool or self.dynamic_bool:
                    total_batch_loss += self.hyperparameters["alpha"] * data_loss
                else:
                    total_batch_loss += data_loss

            outputs = torch.cat(outputs, dim=1)    
            if self.monotonicity_bool:
                mono_loss = monotone_penalty(outputs,
                                               [0, 2, 3, 4, 5, 6],
                                               mask_target)
                mono_loss *= 1e3 # Scale the penalty
                total_batch_loss += epoch_MonoParam * mono_loss

            if self.static_bool:
                biomass_loss = biomass_penalty(outputs,
                                               [2, 3, 4, 5],
                                               y_means=self.target_ts_mean,
                                               y_std=self.target_ts_std,
                                               mask=mask_target)
                assimilation_loss = assimilation_penalty(outputs,
                                                         [8, 9, 10],
                                                         y_means=self.target_ts_mean,
                                                         y_std=self.target_ts_std,
                                                         mask=mask_target)
                total_batch_loss += self.hyperparameters["beta"] * (biomass_loss + assimilation_loss)
            
            if self.dynamic_bool:
                dry_matter_increase_loss = dry_matter_increase_penalty(outputs,
                                                                      [2, 6, 7],
                                                                      y_means=self.target_ts_mean,
                                                                      y_std=self.target_ts_std,
                                                                      mask=mask_target)
                total_batch_loss += self.hyperparameters["gamma"] * dry_matter_increase_loss


            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                           self.hyperparameters["max_norm"])
            self.optimizer.step()
            self.scheduler.step()

            total_epoch_loss += total_batch_loss.item()
            total_epoch_data_loss += total_batch_data_loss.item()
            if self.monotonicity_bool:
                total_epoch_mono_loss += mono_loss.item()
            if self.static_bool:
                total_epoch_biomass_loss += biomass_loss.item()
                total_epoch_assimilation_loss += assimilation_loss.item()
            if self.dynamic_bool:
                total_epoch_dry_matter_increase_loss += dry_matter_increase_loss.item()

        print(f"TRAIN : Epoch [{epoch+1}/{num_epochs}], Loss: {total_epoch_loss:.4f}")

        checkpoint = {
            "epoch": epoch+1,
            "state_encoder_dict": self.encoder.state_dict(),
            "state_decoder_dict": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            }
        save_checkpoint(checkpoint, filename=f"checkpoints/{self.exp_name}/checkpoint.pth")
        tf_logs = {
            "DD_Loss": total_epoch_data_loss,
            "Loss": total_epoch_loss,
            "LR": self.optimizer.param_groups[0]['lr'],
            "TF_ratio": tf_ratio,
        }

        if self.monotonicity_bool:
            tf_logs["MonotonicityLoss"] = total_epoch_mono_loss
            tf_logs["Mono_param"] = epoch_MonoParam

        if self.static_bool:
            tf_logs["BiomassLoss"] = total_epoch_biomass_loss
            tf_logs["AssimilationLoss"] = total_epoch_assimilation_loss
        
        if self.dynamic_bool:
            tf_logs["DryMatterIncreaseLoss"] = total_epoch_dry_matter_increase_loss
        self.writer.log_epoch(epoch, "Train", tf_logs)

    def eval_epoch(self,
                   epoch):
        total_epoch_data_loss = 0.0
        total_epoch_data_mae = 0.0
        total_epoch_mono_loss = 0.0
        total_epoch_biomass_loss = 0.0
        total_epoch_assimilation_loss = 0.0
        total_epoch_dry_matter_increase_loss = 0.0

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                total_batch_data_mae = 0.0
                total_batch_data_loss = 0.0
                mono_loss = 0.0
                biomass_loss = 0.0
                assimilation_loss = 0.0
                dry_matter_increase_loss = 0.0
                outputs = []

                static_data_cat = batch["static_data_cat"].to(self.device)
                static_data_num = batch["static_data_num"].to(self.device)
                before_ts = batch["before_ts"].to(self.device)
                after_ts = batch["after_ts"].to(self.device)
                target_ts = batch["target_ts"].to(self.device)
                mask_target = batch["mask_target"].to(self.device)

                latent, x_t = self.encoder(static_data_num, static_data_cat, before_ts)
                h_t = latent
                for t in range(after_ts.shape[1]):
                    output, h_t, h_output = self.decoder(x_t.unsqueeze(1), h_t, after_ts[:, t, :])
                    outputs.append(output.unsqueeze(1))
                    x_t = output

                    mask_t = mask_target[:, t]

                    loss_unreduced = self.criterion_mse(output, target_ts[:, t, :])
                    loss_masked = loss_unreduced * mask_t
                    data_loss = loss_masked.sum() / (mask_t.sum() + 1e-8)

                    mae_unreduced = (output - target_ts[:, t, :]).abs()
                    mae_masked = mae_unreduced * mask_t
                    mae_loss = mae_masked.sum() / (mask_t.sum() + 1e-8)

                    total_batch_data_mae += mae_loss
                    total_batch_data_loss += data_loss

                outputs = torch.cat(outputs, dim=1)

                mono_loss = monotone_penalty(outputs,
                                               [0, 2, 3, 4, 5, 6],
                                               mask_target)
                mono_loss *= 1e3 # Scale the penalty

                biomass_loss = biomass_penalty(outputs,
                                               [2, 3, 4, 5],
                                               y_means=self.target_ts_mean,
                                               y_std=self.target_ts_std,
                                               mask=mask_target)

                assimilation_loss = assimilation_penalty(outputs,
                                                         [8, 9, 10],
                                                         y_means=self.target_ts_mean,
                                                         y_std=self.target_ts_std,
                                                         mask=mask_target)

                dry_matter_increase_loss = dry_matter_increase_penalty(outputs,
                                                                      [2, 6, 7],
                                                                      y_means=self.target_ts_mean,
                                                                      y_std=self.target_ts_std,
                                                                      mask=mask_target)

                total_epoch_mono_loss += mono_loss.item()
                total_epoch_data_loss += total_batch_data_loss.item()
                total_epoch_biomass_loss += biomass_loss.item()
                total_epoch_assimilation_loss += assimilation_loss.item()
                total_epoch_dry_matter_increase_loss += dry_matter_increase_loss.item()
                total_epoch_data_mae += total_batch_data_mae.item()

            tf_logs = {
                "DD_Loss": total_epoch_data_loss,
                "MonotonicityLoss": total_epoch_mono_loss,
                "BiomassLoss": total_epoch_biomass_loss,
                "AssimilationLoss": total_epoch_assimilation_loss,
                "DryMatterIncreaseLoss": total_epoch_dry_matter_increase_loss,
                "MAE": total_epoch_data_mae
            }
            
            self.writer.log_epoch(epoch, "Eval", tf_logs)

            if total_epoch_data_loss < self.best_val_loss:
                self.best_val_loss = total_epoch_data_loss
                checkpoint = {
                    "epoch": epoch+1,
                    "state_encoder_dict": self.encoder.state_dict(),
                    "state_decoder_dict": self.decoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_checkpoint(checkpoint,
                                filename=f"checkpoints/{self.exp_name}/best_model.pth",
                                best_flag=True)
    

    def train_loop(self):
            self.criterion_mse = torch.nn.MSELoss(reduction='none')
            self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
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
                    print(self.new_batch["id"][:3])
                self.train_epoch(epoch, self.hyperparameters["num_epochs"])
                if not self.overfit:
                    self.eval_epoch(epoch)