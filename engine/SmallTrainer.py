from tqdm import tqdm
import torch
import pickle
from utils.schedulers import teacher_forcing_decay
from utils.losses import monotone_penalty, biomass_penalty, assimilation_penalty, dry_matter_increase_penalty
from utils.utilities import save_checkpoint
from utils.tensorflowLogger import EpochWriter
from utils.mlflowLogger import LoggerMLflow

class Trainer(object):
    def __init__(self,
                 exp_name,
                 encoder,
                 decoder,
                 learning_rate,
                 num_epochs,
                 teacher_forcing_ratio,
                 teacher_forcing_bool,
                 clip_grad_max_norm,
                 optimizer,
                 scheduler,
                 train_dataloader,
                 val_dataloader,
                 checkpoints_path,
                 logs_path,
                 monotonicity_bool,
                 means_std_path,
                 device,
                 mlflow_bool=False,
                 overfit=False):

        # Core components
        self.exp_name = exp_name
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.writer = EpochWriter(exp_name)
        self.mlflow_logger = LoggerMLflow(mlflow_bool)

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
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.teacher_forcing_bool = teacher_forcing_bool
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_max_norm = clip_grad_max_norm
        self.monotonicity_bool = monotonicity_bool
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
        if self.teacher_forcing_bool:
            tf_ratio = teacher_forcing_decay(epoch, num_epochs, end_ratio=self.teacher_forcing_ratio)
        else:
            tf_ratio = 0.0
        total_epoch_loss = 0.0
        total_epoch_data_loss = 0.0
        total_epoch_biomass_loss = 0.0
        total_epoch_assimilation_loss = 0.0
        total_epoch_dry_matter_increase_loss = 0.0

        for batch in tqdm(self.train_dataloader):
            total_batch_data_loss = 0.0
            total_batch_loss = 0.0
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
            x = torch.cat([x_t.unsqueeze(1), target_ts[:, :-1, :]], dim=1)
            h_0 = latent  # h_0
            outputs, _ = self.decoder(x, h_0, after_ts, ar=False)
            loss_unreduced = self.criterion_mse(outputs, target_ts)
            loss_masked = loss_unreduced * mask_target
            data_loss = loss_masked.sum() / (mask_target.sum() + 1e-8)
            # Just a conciense reminder:
            # Loss is additioned over all samples, time steps and features, then averaged over samples and time steps (masked).
            # In this way, The final loss is the mean sum over features.
            total_batch_loss = data_loss
            total_batch_data_loss = data_loss


            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                           self.clip_grad_max_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_epoch_loss += total_batch_loss.item()
            total_epoch_data_loss += total_batch_data_loss

        print(f"TRAIN : Epoch [{epoch+1}/{num_epochs}], Loss: {total_epoch_loss:.4f}")

        ckpt_path = f"{self.checkpoints_path}/checkpoint.pth"
        checkpoint = {
            "epoch": epoch+1,
            "state_encoder_dict": self.encoder.state_dict(),
            "state_decoder_dict": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            }
        save_checkpoint(checkpoint, filename=ckpt_path)
        self.mlflow_logger.mlflow_log_checkpoint(ckpt_path)

        tf_logs = {
            "DD_Loss": total_epoch_data_loss,
            "Loss": total_epoch_loss,
            "LR": self.optimizer.param_groups[0]['lr'],
            "TF_ratio": tf_ratio,
        }
        self.writer.log_epoch(epoch, "Train", tf_logs)
        self.mlflow_logger.mlflow_log_metrics(tf_logs, step=epoch, prefix="Train")


    def eval_epoch(self,
                   epoch):
        total_epoch_data_loss = 0.0
        total_epoch_data_mae = 0.0
        total_epoch_mono_loss = 0.0

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                total_batch_data_mae = 0.0
                total_batch_data_loss = 0.0
                mono_loss = 0.0
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
                    output, h_t, h_output = self.decoder(x_t.unsqueeze(1),
                                                         h_t,
                                                         after_ts[:, t, :],
                                                         ar=True)
                    outputs.append(output.unsqueeze(1))
                    x_t = output

                outputs = torch.cat(outputs, dim=1)

                loss_unreduced = self.criterion_mse(outputs, target_ts)
                loss_masked = loss_unreduced * mask_target
                total_batch_data_loss = loss_masked.sum() / (mask_target.sum() + 1e-8)

                mae_unreduced = (outputs - target_ts).abs()
                mae_masked = mae_unreduced * mask_target
                total_batch_data_mae = mae_masked.sum() / (mask_target.sum() + 1e-8)

                mono_loss = monotone_penalty(outputs,
                                             [0, 2, 3],
                                             mask_target)

                total_epoch_mono_loss += mono_loss.item()
                total_epoch_data_loss += total_batch_data_loss.item()
                total_epoch_data_mae += total_batch_data_mae.item()

            tf_logs = {
                "DD_Loss": total_epoch_data_loss,
                "MonotonicityLoss": total_epoch_mono_loss,
                "MAE": total_epoch_data_mae
            }
            self.writer.log_epoch(epoch, "Eval", tf_logs)
            self.mlflow_logger.mlflow_log_metrics(tf_logs, step=epoch, prefix="Eval")

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
                self.mlflow_logger.mlflow_log_checkpoint(path=f"checkpoints/{self.exp_name}/best_model.pth",
                                                         artifact_path="best_model")

    def train_loop(self):
            self.criterion_mse = torch.nn.MSELoss(reduction='none')

            for epoch in range(self.num_epochs):
                self.train_epoch(epoch, self.num_epochs)
                self.eval_epoch(epoch)