import json
from pathlib import Path

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, Dataset_PHM
from torch.utils.data import DataLoader, Subset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'phm': Dataset_PHM,

}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    elif 'phm' in args.data:
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            seq_len=args.seq_len,
            stride=getattr(args, "dataset_window_stride", None) or getattr(args, "stride", 512),
            cache_dir=getattr(args, "phm_cache_dir", "./cache_data"),
            split_metadata_path=getattr(args, "phm_split_metadata", None),
            allow_legacy_split=getattr(args, "allow_legacy_phm_split", False),
        )
        subset_metadata = getattr(args, "phm_balanced_subset_metadata", "")
        if subset_metadata:
            subset_metadata_path = Path(subset_metadata)
            if not subset_metadata_path.exists():
                raise FileNotFoundError(f"Balanced subset metadata not found: {subset_metadata_path}")
            subset = json.loads(subset_metadata_path.read_text())
            if subset.get("clean_split_metadata") != getattr(args, "phm_split_metadata", None):
                raise RuntimeError(
                    "Balanced subset clean split metadata mismatch: "
                    f"{subset.get('clean_split_metadata')} != {getattr(args, 'phm_split_metadata', None)}"
                )
            split_payload = subset.get("splits", {}).get(flag)
            if not split_payload:
                raise RuntimeError(f"Balanced subset metadata does not contain split={flag}")
            positions = split_payload.get("positions")
            if not positions:
                raise RuntimeError(f"Balanced subset split={flag} has empty positions")
            data_set = Subset(data_set, positions)


    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
