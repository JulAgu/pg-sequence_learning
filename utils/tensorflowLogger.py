from torch.utils.tensorboard import SummaryWriter

class EpochWriter(object):
    def __init__(self,
                 exp_name,
                 log_dir="TF_logs",
                 ):
        
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{exp_name}")
        
    def log_epoch(self,
                  epoch,
                  tag,
                  dico):

        for key, value in dico.items():
            self.writer.add_scalar(f"{tag}/{key}", value, epoch)

