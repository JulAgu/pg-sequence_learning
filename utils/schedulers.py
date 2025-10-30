import math

def cosine_scheduler(current_epoch,
                     total_epoch,
                     beta_max):
    start_step = total_epoch / 2
    if current_epoch < start_step:
        return 0.0
    
    progress = (current_epoch - start_step) / (total_epoch - start_step)
    progress = min(progress, 1.0)
    return beta_max * (1 - math.cos(math.pi * progress)) / 2


def teacher_forcing_decay(current_epoch,
                          total_epochs,
                          start_ratio=1.0,
                          end_ratio=0.5):
    ratio = end_ratio + (start_ratio - end_ratio) * (1 - current_epoch / total_epochs)
    return ratio