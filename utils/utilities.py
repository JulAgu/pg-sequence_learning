import torch

def save_checkpoint(state,
                    filename="checkpoint.pth.tar",
                    best_flag=False,
                    ):
    
    torch.save(state, filename)

    if best_flag:
        print("==== Saving new best model ====")
    else:
        print("==== Saving checkpoint ====")


def load_checkpoint(checkpoint,
                    encoder,
                    decoder,
                    optimizer,
                    stop_branch=None,
                    stop_optimizer=None,
                    ):
    
    if encoder is not None:
        encoder.load_state_dict(checkpoint["state_encoder_dict"])
    if decoder is not None:
        decoder.load_state_dict(checkpoint["state_decoder_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if stop_branch is not None:
        stop_branch.load_state_dict(checkpoint["stop_branch"])
    if stop_optimizer is not None:
        stop_optimizer.load_state_dict(checkpoint["stop_optimizer"])

    print("=== Loading checkpoint ===")





