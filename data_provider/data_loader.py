import os
import random
import re
import json
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')


def _short_hash(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]


def _file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))


        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:

            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

def _load_and_parse_single_file(sample_path):

    try:
        df = pd.read_csv(sample_path)
        assert df.shape[1] == 21
        full_data = df.values.astype(np.float32)
        folder_name = sample_path.parent.name
        return sample_path, full_data, folder_name
    except Exception as e:

        return sample_path, None, str(e)


class Dataset_PHM(Dataset):
    def __init__(self, root_path, seq_len=512, stride=512, flag='train',
                 verbose=True, cache_dir='./cache_data', debug=False,
                 split_metadata_path=None, allow_legacy_split=False):
        super().__init__()
        self.root_path = root_path
        self.seq_len = seq_len
        self.stride = stride
        self.flag = flag
        self.verbose = verbose
        self.debug = debug
        self.split_metadata_path = split_metadata_path
        self.allow_legacy_split = allow_legacy_split
        self.split_metadata = None
        self.source_record_by_file_index = {}
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"phm_cache_optimized_v2_seq{seq_len}_stride{stride}.pt")
        self.all_file_data = []
        self.window_pointers = []
        self.labels = {}

        if os.path.exists(self.cache_path):
            self._log(f" loding_from_cache: {self.cache_path}")
            cache_data = torch.load(self.cache_path, weights_only=False)
            if cache_data.get("root_path") and cache_data["root_path"] != str(Path(self.root_path).resolve()):
                raise RuntimeError("PHM cache belongs to a different dataset root")
            self.all_file_data = cache_data["all_file_data"]
            self.window_pointers = cache_data["window_pointers"]
            self.labels = cache_data["labels"]
        else:
            self._index_and_load_dataset()
            torch.save({
                "all_file_data": self.all_file_data,
                "window_pointers": self.window_pointers,
                "labels": self.labels,
            }, self.cache_path)
            self._log(f" cachedata_saved_in: {self.cache_path}")

        if self.split_metadata_path:
            self.indices = self._load_v3_split(self.split_metadata_path)
        elif self.allow_legacy_split:
            self.indices = self._load_or_create_legacy_split(cache_dir)
        else:
            raise RuntimeError(
                "Dataset_PHM requires an explicit root/stride-aware split metadata file. "
                "Pass --phm_split_metadata pointing to a splits_v3 metadata.json. "
                "Legacy phm_split_*.pt reuse is disabled by default to prevent split/cache mismatch."
            )

        self._validate_indices()
        self._log(f" use split={self.flag}, windows={len(self.indices)}")

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _load_v3_split(self, metadata_path):
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Split metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        root_resolved = str(Path(self.root_path).resolve())
        expected = {
            "root_hash": _short_hash(root_resolved),
            "seq_len": self.seq_len,
            "window_stride": self.stride,
            "num_windows": len(self.window_pointers),
        }
        if metadata.get("root_path") != root_resolved:
            raise RuntimeError(f"Split metadata root_path mismatch: {metadata.get('root_path')} != {root_resolved}")
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"Split metadata {key} mismatch: {metadata.get(key)} != {value}")
        if metadata.get("cache_path") and str(Path(metadata["cache_path"]).resolve()) != str(Path(self.cache_path).resolve()):
            raise RuntimeError(f"Split metadata cache_path mismatch: {metadata.get('cache_path')} != {self.cache_path}")
        split_files = metadata.get("split_files", {})
        split_path = split_files.get(self.flag)
        if not split_path:
            raise RuntimeError(f"Split metadata does not contain split file for flag={self.flag}")
        split_path = Path(split_path)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        indices = torch.load(split_path, weights_only=False)
        self.split_index_path = str(split_path)
        self.split_metadata = metadata
        self.source_record_by_file_index = {
            int(record["file_index"]): record
            for record in metadata.get("source_records", [])
            if "file_index" in record
        }
        self._log(
            " split from v3 metadata: "
            f"flag={self.flag}, split_path={split_path}, root_hash={metadata.get('root_hash')}, "
            f"seq_len={metadata.get('seq_len')}, window_stride={metadata.get('window_stride')}, "
            f"split_mode={metadata.get('split_mode')}, seed={metadata.get('seed')}, "
            f"num_windows={metadata.get('num_windows')}, counts={metadata.get('audit', {}).get('window_counts')}, "
            f"source_overlap={metadata.get('audit', {}).get('source_overlap_counts')}"
        )
        return list(indices)

    def _load_or_create_legacy_split(self, cache_dir):
        self.split_index_path = os.path.join(cache_dir, f"phm_split_{self.flag}.pt")
        if os.path.exists(self.split_index_path):
            self._log(f" legacy split from cache: {self.split_index_path}")
            return torch.load(self.split_index_path, weights_only=False)

        self._log("creating legacy window-level random split train/val/test")
        all_indices = list(range(len(self.window_pointers)))
        random.seed(42)
        random.shuffle(all_indices)
        total = len(all_indices)
        train_ratio, val_ratio = 0.7, 0.15
        split_dict = {
            'train': all_indices[:int(total * train_ratio)],
            'val': all_indices[int(total * train_ratio):int(total * (train_ratio + val_ratio))],
            'test': all_indices[int(total * (train_ratio + val_ratio)):]
        }
        for k, idxs in split_dict.items():
            torch.save(idxs, os.path.join(cache_dir, f"phm_split_{k}.pt"))
        return split_dict[self.flag]

    def _validate_indices(self):
        if not self.indices:
            raise RuntimeError(f"Empty split indices for flag={self.flag}")
        max_idx = max(self.indices)
        min_idx = min(self.indices)
        if min_idx < 0 or max_idx >= len(self.window_pointers):
            raise RuntimeError(
                f"Split indices out of range for flag={self.flag}: "
                f"min={min_idx}, max={max_idx}, len(window_pointers)={len(self.window_pointers)}"
            )

    def _parse_label_from_dirname(self, dirname):
        label = [0] * 21
        components = dirname.split("_")
        for comp in components:
            sub_comps = comp.split("+")
            for sub in sub_comps:
                if sub.startswith("M"):
                    idx = int(sub[1:])  # M0-M4: indices 0-4
                    label[idx] = 1
                elif sub.startswith("G"):
                    idx = int(sub[1:])  # G0-G8: indices 5-13
                    label[5 + idx] = 1
                elif sub.startswith("LA"):
                    match = re.match(r"LA(\d+)", sub)
                    if match:
                        idx = int(match.group(1))  # LA0-LA4: indices 14-18
                        label[14 + idx] = 1
                elif sub.startswith("RA"):
                    match = re.match(r"RA(\d+)", sub)
                    if match:
                        idx = int(match.group(1))  # RA0-RA1: indices 19-20
                        label[19 + idx] = 1
        return label


    def _load_full_data(self, sample_file_path):
        try:
            df = pd.read_csv(sample_file_path)
            assert df.shape[1] == 21, f"{sample_file_path} has wrong shape"
            return df.values.astype(np.float32)  # shape: [T, 21]
        except Exception as e:
            raise RuntimeError(f"Cannot load {sample_file_path}: {e}") from e

    def _index_and_load_dataset(self):


        # find all files
        all_csv_files = sorted(Path(self.root_path).rglob('*.csv'))
        self._log(f"Loading {len(all_csv_files)} CSV files into memory")


        results = []
        with ProcessPoolExecutor() as executor:

            future_to_path = {executor.submit(_load_and_parse_single_file, path): path for path in all_csv_files}


            pbar = tqdm(as_completed(future_to_path), total=len(all_csv_files), desc="Parallel Pre-loading")
            for future in pbar:
                path = future_to_path[future]
                try:

                    results.append(future.result())
                except Exception as exc:
                    self._log(f'\n[ERROR] at {path} ,reason is {exc}')


        self._log("set all dataset")


        results.sort(key=lambda x: x[0])

        file_index_counter = 0
        total_windows = 0

        for sample_path, full_data, folder_name_or_error in tqdm(results, desc="Indexing windows"):
            if full_data is None:
                self._log(f"[WARNING] Skipping {sample_path}: {folder_name_or_error}")
                continue


            self.all_file_data.append(full_data)
            folder_name = folder_name_or_error
            if folder_name not in self.labels:
                self.labels[folder_name] = self._parse_label_from_dirname(folder_name)


            total_len = full_data.shape[0]
            for start in range(0, total_len - self.seq_len + 1, self.stride):
                self.window_pointers.append((file_index_counter, start, folder_name))
                total_windows += 1

            file_index_counter += 1

        self._log(f"finish loading data,total windows : {total_windows} ")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]
        file_idx, start_idx, folder_name = self.window_pointers[real_idx]


        full_data = self.all_file_data[file_idx]
        window = full_data[start_idx: start_idx + self.seq_len]


        if window.shape[0] < self.seq_len:
            pad_len = self.seq_len - window.shape[0]
            window = np.pad(window, ((0, pad_len), (0, 0)), mode="constant")


        data_tensor = torch.from_numpy(window)
        label_tensor = torch.tensor(self.labels[folder_name], dtype=torch.float)


        name = f"{folder_name}/{start_idx}"
        source_record = self.source_record_by_file_index.get(file_idx, {})
        source_path = str(source_record.get("csv_path", ""))
        source_id = source_path or f"file_index:{file_idx}"
        condition_id = int(source_record.get("condition_id", -1))
        sample_id = f"{source_id}::window_start={start_idx}"
        valid_len = min(len(full_data) - start_idx, self.seq_len)
        padding_mask_tensor = torch.zeros(self.seq_len, dtype=torch.bool)
        padding_mask_tensor[:valid_len] = 1

        return {
            "data": data_tensor,
            "label": label_tensor,
            "name": name,
            "sample_id": sample_id,
            "source_id": source_id,
            "source_file": source_path,
            "condition_id": condition_id,
            "window_start": int(start_idx),
            "window_index": int(real_idx),
            "label_dir": folder_name,
            "padding_mask": padding_mask_tensor
        }
