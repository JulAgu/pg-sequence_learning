import torch

def monotone_penalty(y_hat,
                     index_list,
                     mask):
    # Differences between consecutive steps
    diff = y_hat[:, :-1, index_list] - y_hat[:, 1:, index_list]  # [B, T-1]
    diff = torch.relu(diff)
    diff_masked = diff * mask[:,1:]
    penalty = diff_masked.sum() / (mask.sum() + 1e-8)
    return penalty


def dry_matter_increase_penalty(y_hat,
                                index_list,
                                y_means,
                                y_std,
                                mask):
    """
    Expected index order in the index_list is: [TAGP, TWRT, DMI]

    This function calculates a penalty on the deviations from the equality : DMI_{t-1} = (TAGP_{t}-TAGP_{t-1})+(TWRT{t}-TWRT{t-1})
    """
    mask = mask.squeeze(2)
    y_means = y_means[index_list]
    y_std = y_std[index_list]
    y_hat = y_hat[:, :, index_list]

    original_y_hat = y_hat * y_std + y_means
    prev_y_hat = original_y_hat[:, :-1, :]  # Get the previous time step
    curr_y_hat = original_y_hat[:, 1:, :]  # Get the current time step

    # Compute the dry matter increase penalty
    tagp_diff = curr_y_hat[:, :, 0] - prev_y_hat[:, :, 0]
    twrt_diff = curr_y_hat[:, :, 1] - prev_y_hat[:, :, 1]
    inconsistence = torch.square(prev_y_hat[:, :, 2] - (tagp_diff + twrt_diff))
    constraint_std = torch.sqrt(y_std[0]**2 + y_std[1]**2 + y_std[2]**2)
    inconsistence = inconsistence / constraint_std
    penalty = (inconsistence * mask[:,1:]).sum() / (mask.sum() + 1e-8)
    return penalty


def biomass_penalty(y_hat,
                    index_list,
                    y_means,
                    y_std,
                    mask):
    '''
    Expected index order in the index_list is: [TAGP, TWSO, TMLV, TWST]

    This function calculates a penalty on the deviations from the equality : TAGP = TWSO + TMLV + TWST
    '''
    mask = mask.squeeze(2)
    y_means = y_means[index_list]
    y_std = y_std[index_list]
    y_hat = y_hat[:, :, index_list]

    original_y_hat = y_hat * y_std + y_means
    total_biomass = original_y_hat[:, :, 0]
    partial_biomass = original_y_hat[:, :, 1:]
    # Compute the biomass penalty
    inconsistence = torch.square(total_biomass - partial_biomass.sum(dim=-1))
    constraint_std = torch.sqrt(y_std[0]**2 + y_std[1]**2 + y_std[2]**2 + y_std[3]**2)
    inconsistence = inconsistence / constraint_std
    penalty = (inconsistence * mask).sum() / (mask.sum() + 1e-8)
    return penalty


def assimilation_penalty(y_hat,
                         index_list,
                         y_means,
                         y_std,
                         mask):
    '''
    Expected index order in the index_list is: [ASRC, GASS, MRES]

    This function calculates a penalty on the deviations from the equality : ASRC = GASS - MRES
    '''
    mask = mask.squeeze(2)
    y_means = y_means[index_list]
    y_std = y_std[index_list]
    y_hat = y_hat[:, :, index_list]

    original_y_hat = y_hat * y_std + y_means
    asrc = original_y_hat[:, :, 0]
    gass = original_y_hat[:, :, 1]
    mres = original_y_hat[:, :, 2]
    # Compute the biomass penalty
    inconsistence = torch.square(asrc - (gass - mres))
    constraint_std = torch.sqrt(y_std[0]**2 + y_std[1]**2 + y_std[2]**2)
    inconsistence = inconsistence / constraint_std
    penalty = (inconsistence * mask).sum() / (mask.sum() + 1e-8)
    return penalty
