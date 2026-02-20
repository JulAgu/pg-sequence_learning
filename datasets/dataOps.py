import pickle
import pandas as pd
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

    val_dataset = DataSet(ids[train_size:train_size + val_size],
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

    test_dataset = DataSet(ids[train_size + val_size:],
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


def create_ood_datasets(ids, static_data,
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

    train_val_ood = pd.read_csv(raw_data_folder + "entire_ood_train.csv")
    test_ood = pd.read_csv(raw_data_folder + "entire_ood_test.csv")


    train_val_ood_ids = set(train_val_ood["id"].tolist())
    test_ood_ids = set(test_ood["id"].tolist())

    # indices for test OOD (order preserved)
    test_ood_ids_idx = [
        i for i, id_ in enumerate(ids) if id_ in test_ood_ids
    ]

    # IID ids (order preserved)
    iid_ids = [
        id_ for id_ in ids
        if id_ not in train_val_ood_ids and id_ not in test_ood_ids
    ]

    # indices for IID ids
    iid_ids_idx = [
        i for i, id_ in enumerate(ids)
        if id_ not in train_val_ood_ids and id_ not in test_ood_ids
    ]


    # train_val_ood_ids = set(train_val_ood["id"].tolist())
    # test_ood_ids = set(test_ood["id"].tolist())
    # test_ood_ids_idx = [i for i, id_ in enumerate(ids) if id_ in test_ood_ids]

    # iid_ids = list(set(ids) - train_val_ood_ids - test_ood_ids)
    # iid_ids_set = set(iid_ids)
    # iid_ids_idx = [i for i, id_ in enumerate(ids) if id_ in iid_ids_set]
    
    iid_static_data = static_data[iid_ids_idx]
    iid_before_ts = before_ts[iid_ids_idx]
    iid_after_ts = after_ts[iid_ids_idx]
    iid_target_ts = target_ts[iid_ids_idx]
    iid_mask_target = mask_target[iid_ids_idx]

    total_size = len(iid_static_data)
    train_size = int(total_size * train_size)
    val_size = int(total_size * val_size)
    test_size = total_size - train_size - val_size

    DataSet = globals()[type_of_dataset]

    train_dataset = DataSet(iid_ids[:train_size],
                            iid_static_data[:train_size],
                            iid_before_ts[:train_size],
                            iid_after_ts[:train_size],
                            iid_target_ts[:train_size],
                            iid_mask_target[:train_size],
                            target_mode=target_mode,
                            means_and_stds_path=means_and_stds_path,
                            entire_bool=entire_bool,
                            )

    with open(means_and_stds_path, "rb") as f:
        means_and_stds_dict = pickle.load(f)

    val_dataset = DataSet(iid_ids[train_size:train_size + val_size],
                          iid_static_data[train_size:train_size + val_size],
                          iid_before_ts[train_size:train_size + val_size],
                          iid_after_ts[train_size:train_size + val_size],
                          iid_target_ts[train_size:train_size + val_size],
                          iid_mask_target[train_size:train_size + val_size],
                          target_mode=target_mode,
                          means_and_stds_path=means_and_stds_path,
                          means_and_stds_dict=means_and_stds_dict,
                          entire_bool=entire_bool,
                          )

    test_dataset = DataSet(iid_ids[train_size + val_size:],
                           iid_static_data[train_size + val_size:],
                           iid_before_ts[train_size + val_size:],
                           iid_after_ts[train_size + val_size:],
                           iid_target_ts[train_size + val_size:],
                           iid_mask_target[train_size + val_size:],
                           target_mode=target_mode,
                           means_and_stds_path=means_and_stds_path,
                           means_and_stds_dict=means_and_stds_dict,
                           entire_bool=entire_bool,
                           )
    
    test_ood_dataset = DataSet(list(test_ood_ids),
                               static_data[test_ood_ids_idx],
                               before_ts[test_ood_ids_idx],
                               after_ts[test_ood_ids_idx],
                               target_ts[test_ood_ids_idx],
                               mask_target[test_ood_ids_idx],
                               target_mode=target_mode,
                               means_and_stds_path=means_and_stds_path,
                               means_and_stds_dict=means_and_stds_dict,
                               entire_bool=entire_bool,
                               )

    print(f"""
          Train_DS = {train_size} obs
          Val_DS = {val_size} obs
          Test_DS = {test_size} obs
          OOD_Test_DS = {len(test_ood_ids)} obs
          """)

    return train_dataset, val_dataset, test_dataset, test_ood_dataset

def create_loc_ood_datasets(ids, static_data,
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
    
    test_ood = pd.read_csv(raw_data_folder + "only_coordinates_ood.csv")

    test_ood_ids = set(test_ood["id"].tolist())
    test_ood_ids_idx = [i for i, id_ in enumerate(ids) if id_ in test_ood_ids]

    iid_ids = list(set(ids) - test_ood_ids)
    iid_ids_set = set(iid_ids)
    iid_ids_idx = [i for i, id_ in enumerate(ids) if id_ in iid_ids_set]
    
    iid_static_data = static_data[iid_ids_idx]
    iid_before_ts = before_ts[iid_ids_idx]
    iid_after_ts = after_ts[iid_ids_idx]
    iid_target_ts = target_ts[iid_ids_idx]
    iid_mask_target = mask_target[iid_ids_idx]

    total_size = len(iid_static_data)
    train_size = int(total_size * train_size)
    val_size = int(total_size * val_size)
    test_size = total_size - train_size - val_size

    DataSet = globals()[type_of_dataset]

    train_dataset = DataSet(iid_ids[:train_size],
                            iid_static_data[:train_size],
                            iid_before_ts[:train_size],
                            iid_after_ts[:train_size],
                            iid_target_ts[:train_size],
                            iid_mask_target[:train_size],
                            target_mode=target_mode,
                            means_and_stds_path=means_and_stds_path,
                            entire_bool=entire_bool,
                            )

    with open(means_and_stds_path, "rb") as f:
        means_and_stds_dict = pickle.load(f)

    val_dataset = DataSet(iid_ids[train_size:train_size + val_size],
                          iid_static_data[train_size:train_size + val_size],
                          iid_before_ts[train_size:train_size + val_size],
                          iid_after_ts[train_size:train_size + val_size],
                          iid_target_ts[train_size:train_size + val_size],
                          iid_mask_target[train_size:train_size + val_size],
                          target_mode=target_mode,
                          means_and_stds_path=means_and_stds_path,
                          means_and_stds_dict=means_and_stds_dict,
                          entire_bool=entire_bool,
                          )

    test_dataset = DataSet(iid_ids[train_size + val_size:],
                           iid_static_data[train_size + val_size:],
                           iid_before_ts[train_size + val_size:],
                           iid_after_ts[train_size + val_size:],
                           iid_target_ts[train_size + val_size:],
                           iid_mask_target[train_size + val_size:],
                           target_mode=target_mode,
                           means_and_stds_path=means_and_stds_path,
                           means_and_stds_dict=means_and_stds_dict,
                           entire_bool=entire_bool,
                           )
    
    test_ood_dataset = DataSet(list(test_ood_ids),
                               static_data[test_ood_ids_idx],
                               before_ts[test_ood_ids_idx],
                               after_ts[test_ood_ids_idx],
                               target_ts[test_ood_ids_idx],
                               mask_target[test_ood_ids_idx],
                               target_mode=target_mode,
                               means_and_stds_path=means_and_stds_path,
                               means_and_stds_dict=means_and_stds_dict,
                               entire_bool=entire_bool,
                               )

    print(f"""
          Train_DS = {train_size} obs
          Val_DS = {val_size} obs
          Test_DS = {test_size} obs
          OOD_Test_DS = {len(test_ood_ids)} obs
          """)

    return train_dataset, val_dataset, test_dataset, test_ood_dataset

def create_dataloaders(train_dataset,
                       val_dataset,
                       test_dataset,
                       batch_size=32):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
    
    return train_loader, val_loader, test_loader