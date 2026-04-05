import os
import polars as pl
import argparse


def main(data_path, output_path):
    train_frac = 0.7
    valid_frac = 0.15
    df = pl.read_csv(data_path)  # targets.csv

    scan_ids = df["scan_id"].unique().to_list()

    n_scans = len(scan_ids)
    num_train_scans = int(n_scans * train_frac)
    num_valid_scans = int(n_scans * valid_frac)

    train_scan_ids = df.filter(pl.col("scan_id").is_in(scan_ids[:num_train_scans]))[
        "series_uid"
    ]

    valid_scan_ids = df.filter(
        pl.col("scan_id").is_in(
            scan_ids[num_train_scans : num_train_scans + num_valid_scans]
        )
    )["series_uid"]

    test_scan_ids = df.filter(
        pl.col("scan_id").is_in(scan_ids[num_train_scans + num_valid_scans :])
    )["series_uid"]

    train_ids = pl.DataFrame({"series_uid": train_scan_ids})
    valid_ids = pl.DataFrame({"series_uid": valid_scan_ids})
    test_ids = pl.DataFrame({"series_uid": test_scan_ids})

    os.makedirs(output_path, exist_ok=True)
    train_ids.write_csv(os.path.join(output_path, "train.csv"))
    valid_ids.write_csv(os.path.join(output_path, "valid.csv"))
    test_ids.write_csv(os.path.join(output_path, "test.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    main(args.data_path, args.output_path)
