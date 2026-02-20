import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class AgrialDataset(Dataset):
    def __init__(self,
                 ids,
                 static_data,
                 before_ts,
                 after_ts,
                 target_ts,
                 mask_target,
                 target_mode="ts",
                 means_and_stds_path=None,
                 means_and_stds_dict=None,
                 entire_bool=True):

        self.ids = ids
        self.static_data_cat = static_data[:,[0,1,2,6]]
        self.static_data_num = static_data[:,[3,4,5,7,8]]
        self.before_ts = before_ts
        self.after_ts = after_ts
        if entire_bool:
            self.target_ts = target_ts
        else:
            self.target_ts = target_ts[:,:,:4]
        self.mask_target = mask_target

        if means_and_stds_dict is not None:
            # If means and stds are provided, use them for normalization
            self.static_data_num_mean = means_and_stds_dict["static_data_num_mean"]
            self.static_data_num_std = means_and_stds_dict["static_data_num_std"]
            self.before_ts_mean = means_and_stds_dict["before_ts_mean"]
            self.before_ts_std = means_and_stds_dict["before_ts_std"]
            self.after_ts_mean = means_and_stds_dict["after_ts_mean"]
            self.after_ts_std = means_and_stds_dict["after_ts_std"]
            self.target_ts_mean = means_and_stds_dict["target_ts_mean"]
            self.target_ts_std = means_and_stds_dict["target_ts_std"]
        else:
        # From all numeric features, extract mean and std for normalization
            self.static_data_num_mean = self.static_data_num.mean(0)
            self.static_data_num_std = self.static_data_num.std(0)
            self.before_ts_mean = self.before_ts.reshape(-1, self.before_ts.shape[-1]).mean(0)
            self.before_ts_std = self.before_ts.reshape(-1, self.before_ts.shape[-1]).std(0)
            self.after_ts_mean = self.after_ts.reshape(-1, self.after_ts.shape[-1]).mean(0)
            self.after_ts_std = self.after_ts.reshape(-1, self.after_ts.shape[-1]).std(0)
            self.target_ts_mean = self.target_ts.reshape(-1, self.target_ts.shape[-1]).mean(0)
            self.target_ts_std = self.target_ts.reshape(-1, self.target_ts.shape[-1]).std(0)
            self.dic_means_and_stds = {
                "static_data_num_mean": self.static_data_num_mean,
                "static_data_num_std": self.static_data_num_std,
                "before_ts_mean": self.before_ts_mean,
                "before_ts_std": self.before_ts_std,
                "after_ts_mean": self.after_ts_mean,
                "after_ts_std": self.after_ts_std,
                "target_ts_mean": self.target_ts_mean,
                "target_ts_std": self.target_ts_std
            }
            with open(means_and_stds_path, "wb") as f:
                pickle.dump(self.dic_means_and_stds, f)

        # Normalize numeric features
        self.static_data_num = (self.static_data_num - self.static_data_num_mean) / self.static_data_num_std
        self.before_ts = (self.before_ts - self.before_ts_mean) / self.before_ts_std
        self.after_ts = (self.after_ts - self.after_ts_mean) / self.after_ts_std
        self.target_ts = (self.target_ts - self.target_ts_mean) / self.target_ts_std
 
    def __len__(self):
        return len(self.static_data_num)
    
    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "static_data_cat": torch.tensor(self.static_data_cat[idx], dtype=torch.long),
            "static_data_num": torch.tensor(self.static_data_num[idx], dtype=torch.float32),
            "before_ts": torch.tensor(self.before_ts[idx], dtype=torch.float32),
            "after_ts": torch.tensor(self.after_ts[idx], dtype=torch.float32),
            "target_ts": torch.tensor(self.target_ts[idx], dtype=torch.float32),
            "mask_target": torch.tensor(self.mask_target[idx], dtype=torch.int8)
        }


class AgrialStaticDataset(Dataset):
    def __init__(self,
                 ids,
                 static_data,
                 before_ts,
                 after_ts,
                 target_ts,
                 mask_target,
                 target_mode="ts",
                 means_and_stds_path = None,
                 means_and_stds_dict=None,
                 entire_bool=True):

        self.ids = ids
        self.static_data_cat = static_data[:,[0,1,2,6]]
        static_data_num = static_data[:,[3,4,5,7,8]]
        meteo_ts = np.concatenate((before_ts.reshape(before_ts.shape[0], -1), after_ts.reshape(after_ts.shape[0], -1)), axis=1)
        self.static_data_num = np.concatenate((static_data_num, meteo_ts), axis=1)
        
        last_index_list = (mask_target.sum(axis=1) -1).reshape(-1)
        if target_mode=="yield":
            self.target_ts = target_ts[np.arange(len(last_index_list)),last_index_list, 3]
        elif target_mode=="ts":
            self.target_ts = target_ts[:,:,3].reshape(target_ts.shape[0], -1)            
        self.mask_target = mask_target

        if means_and_stds_dict is not None:
            # If means and stds are provided, use them for normalizations
            self.static_data_num_mean = means_and_stds_dict["static_data_num_mean"]
            self.static_data_num_std = means_and_stds_dict["static_data_num_std"]
        else:
        # From all numeric features, extract mean and std for normalization
            self.static_data_num_mean = self.static_data_num.mean(0)
            self.static_data_num_std = self.static_data_num.std(0)
            self.dic_means_and_stds = {
                "static_data_num_mean": self.static_data_num_mean,
                "static_data_num_std": self.static_data_num_std,
            }
            with open(means_and_stds_path, "wb") as f:
                pickle.dump(self.dic_means_and_stds, f)

        # Normalize numeric features
        self.static_data_num = (self.static_data_num - self.static_data_num_mean) / self.static_data_num_std

        # print shapes
        print(f"static_data_cat shape: {self.static_data_cat.shape}")
        print(f"static_data_num shape: {self.static_data_num.shape}")
        print(f"target_ts shape: {self.target_ts.shape}")
 
    def __len__(self):
        return len(self.target_ts)
    
    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "static_data_cat": torch.tensor(self.static_data_cat[idx], dtype=torch.long),
            "static_data_num": torch.tensor(self.static_data_num[idx], dtype=torch.float32),
            "target_ts": torch.tensor(self.target_ts[idx], dtype=torch.float32),
            "mask_target": torch.tensor(self.mask_target[idx], dtype=torch.int8)
        }
    
class AgrialHybridDataset(Dataset):
    def __init__(self,
                 ids,
                 static_data,
                 before_ts,
                 after_ts,
                 target_ts,
                 mask_target,
                 target_mode="ts",
                 means_and_stds_path = None,
                 means_and_stds_dict=None,
                 entire_bool=True):
    
        '''
        The target mode can be "ts", "yield" or "one_ts" 
        '''

        self.ids = ids
        self.static_data_cat = static_data[:,[0,1,2,6]]
        self.static_data_num = static_data[:,[3,4,5,7,8]]
        self.meteo_ts = np.concatenate((before_ts, after_ts), axis=1)
        last_index_list = (mask_target.sum(axis=1) -1).reshape(-1)
        if target_mode=="yield":
            self.target_ts = target_ts[np.arange(len(last_index_list)),last_index_list, 2]
        elif target_mode=="ts":
            self.target_ts = target_ts.reshape(target_ts.shape[0], -1)
        else:
            self.target_ts = target_ts[:,:,2].reshape(target_ts.shape[0], -1)
        self.mask_target = mask_target

        if means_and_stds_dict is not None:
            # If means and stds are provided, use them for normalizations
            self.static_data_num_mean = means_and_stds_dict["static_data_num_mean"]
            self.static_data_num_std = means_and_stds_dict["static_data_num_std"]
            self.meteo_ts_mean = means_and_stds_dict["meteo_ts_mean"]
            self.meteo_ts_std = means_and_stds_dict["meteo_ts_std"]
        else:
        # From all numeric features, extract mean and std for normalization
            self.static_data_num_mean = self.static_data_num.mean(0)
            self.static_data_num_std = self.static_data_num.std(0)
            self.meteo_ts_mean = self.meteo_ts.reshape(-1, self.meteo_ts.shape[-1]).mean(0)
            self.meteo_ts_std = self.meteo_ts.reshape(-1, self.meteo_ts.shape[-1]).std(0)
            self.dic_means_and_stds = {
                "static_data_num_mean": self.static_data_num_mean,
                "static_data_num_std": self.static_data_num_std,
                "meteo_ts_mean": self.meteo_ts_mean,
                "meteo_ts_std": self.meteo_ts_std,
            }
            with open(means_and_stds_path, "wb") as f:
                pickle.dump(self.dic_means_and_stds, f)

        # Normalize numeric features
        self.static_data_num = (self.static_data_num - self.static_data_num_mean) / self.static_data_num_std
        self.meteo_ts = (self.meteo_ts - self.meteo_ts_mean) / self.meteo_ts_std
 
    def __len__(self):
        return len(self.static_data_num)
    
    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "static_data_cat": torch.tensor(self.static_data_cat[idx], dtype=torch.long),
            "static_data_num": torch.tensor(self.static_data_num[idx], dtype=torch.float32),
            "meteo_ts": torch.tensor(self.meteo_ts[idx], dtype=torch.float32),  
            "target_ts": torch.tensor(self.target_ts[idx], dtype=torch.float32),
            "mask_target": torch.tensor(self.mask_target[idx], dtype=torch.int8)
        }