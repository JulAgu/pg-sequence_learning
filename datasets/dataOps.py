import pickle
from torch.utils.data import DataLoader
from datasets.AgrialDS import AgrialDataset, AgrialStaticDataset, AgrialHybridDataset

def create_datasets(ids, static_data,
                    before_ts,
                    after_ts,
                    target_ts,
                    mask_target,
                    train_size=0.8,
                    val_size=0.1,
                    raw_data_folder="data/",
                    means_and_stds_path=None,
                    target_mode=None,
                    type_of_dataset = "AgrialDataset",
                    entire_bool=True):

    total_size = len(static_data)
    train_size = int(total_size * train_size)
    val_size = int(total_size * val_size)
    test_size = total_size - train_size - val_size

    DataSet = globals()[type_of_dataset]

    train_dataset = DataSet(ids[:train_size],
                            static_data[:train_size],
                            before_ts[:train_size],
                            after_ts[:train_size],
                            target_ts[:train_size],
                            mask_target[:train_size],
                            target_mode=target_mode,
                            means_and_stds_path=means_and_stds_path,
                            entire_bool=entire_bool,
                            )

    with open(means_and_stds_path, "rb") as f:
        means_and_stds_dict = pickle.load(f)

    val_dataset = DataSet(ids[:train_size],
                          static_data[train_size:train_size + val_size],
                          before_ts[train_size:train_size + val_size],
                          after_ts[train_size:train_size + val_size],
                          target_ts[train_size:train_size + val_size],
                          mask_target[train_size:train_size + val_size],
                          target_mode=target_mode,
                          means_and_stds_path=means_and_stds_path,
                          means_and_stds_dict=means_and_stds_dict,
                          entire_bool=entire_bool,
                          )

    test_dataset = DataSet(ids[:train_size],
                           static_data[train_size + val_size:],
                           before_ts[train_size + val_size:],
                           after_ts[train_size + val_size:],
                           target_ts[train_size + val_size:],
                           mask_target[train_size + val_size:],
                           target_mode=target_mode,
                           means_and_stds_path=means_and_stds_path,
                           means_and_stds_dict=means_and_stds_dict,
                           entire_bool=entire_bool,
                           )

    print(f"""
          Train_DS = {train_size} obs
          Val_DS = {val_size} obs
          Test_DS = {test_size} obs
          """)

    return train_dataset, val_dataset, test_dataset


def create_ood_datasets(ids,
                        static_data,
                        before_ts,
                        after_ts,
                        target_ts,
                        mask_target,
                        ids_to_eliminate_of_train_validation=None,
                        test_ids=None,
                        train_size=0.8,
                        val_size=0.2,
                        raw_data_folder="data/",
                        means_and_stds_path=None,
                        target_mode=None,
                        type_of_dataset = "AgrialDataset"):
    """
    In this version train_size and val_size should add up to 1.0, because test set is extrictly defined.
    """

    available_ids = set(ids) - set(ids_to_eliminate_of_train_validation)
    available_size = len(available_ids)
    available_indices = [i for i, id_ in enumerate(ids) if id_ in available_ids]
    train_size = int(available_size * train_size)
    val_size = int(available_size * val_size)
    test_size = available_size - train_size - val_size
    if test_ids is not None:
        test_size = len(test_ids)
        test_indices = [i for i, id_ in enumerate(ids) if id_ in test_ids]

    DataSet = globals()[type_of_dataset]

    train_dataset = DataSet([ids[i] for i in available_indices[:train_size]],
                            static_data[available_indices[:train_size]],
                            before_ts[available_indices[:train_size]],
                            after_ts[available_indices[:train_size]],
                            target_ts[available_indices[:train_size]],
                            mask_target[available_indices[:train_size]],
                            target_mode=target_mode,
                            means_and_stds_path=means_and_stds_path
                            )

    with open(means_and_stds_path, "rb") as f:
        means_and_stds_dict = pickle.load(f)

    val_dataset = DataSet([ids[i] for i in available_indices[train_size:train_size + val_size]],
                          static_data[available_indices[train_size:train_size + val_size]],
                          before_ts[available_indices[train_size:train_size + val_size]],
                          after_ts[available_indices[train_size:train_size + val_size]],
                          target_ts[available_indices[train_size:train_size + val_size]],
                          mask_target[available_indices[train_size:train_size + val_size]],
                          target_mode=target_mode,
                          means_and_stds_path=means_and_stds_path,
                          means_and_stds_dict=means_and_stds_dict,
                          )
    if test_ids is not None:
        test_dataset = DataSet([ids[i] for i in test_indices],
                               static_data[test_indices],
                               before_ts[test_indices],
                               after_ts[test_indices],
                               target_ts[test_indices],
                               mask_target[test_indices],
                               target_mode=target_mode,
                               means_and_stds_path=means_and_stds_path,
                               means_and_stds_dict=means_and_stds_dict,
                               )
    else:
        test_dataset = DataSet([ids[i] for i in available_indices[train_size + val_size:]],
                               static_data[available_indices[train_size + val_size:]],
                               before_ts[available_indices[train_size + val_size:]],
                               after_ts[available_indices[train_size + val_size:]],
                               target_ts[available_indices[train_size + val_size:]],
                               mask_target[available_indices[train_size + val_size:]],
                               target_mode=target_mode,
                               means_and_stds_path=means_and_stds_path,
                               means_and_stds_dict=means_and_stds_dict,
                                )

    print(f"""
          Train_DS = {train_size} obs
          Val_DS = {val_size} obs
          Test_DS = {test_size} obs
          """)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset,
                       val_dataset,
                       test_dataset,
                       batch_size=32):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
    
    return train_loader, val_loader, test_loader