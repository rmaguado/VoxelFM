import os
import shutil
import random
import pandas as pd


random.seed(4)

main_split_path = "evaluation/iCTCF/splits"
output_dir = "evaluation/iCTCF/subplits"

os.makedirs(output_dir, exist_ok=True)

fractions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]

train_split_path = os.path.join(main_split_path, "train.csv")
valid_split_path = os.path.join(main_split_path, "valid.csv")
test_split_path = os.path.join(main_split_path, "test.csv")
df = pd.read_csv(train_split_path)

series_uids = df["series_uid"].to_list()
total = len(series_uids)

for fraction in fractions:

    n_subset = int(total * fraction)
    subset = random.sample(series_uids, n_subset)

    split_output_dir = os.path.join(output_dir, f"split_{fraction:.02f}")
    os.makedirs(split_output_dir, exist_ok=True)

    df_frac = pd.DataFrame({"series_uid": subset})
    df_frac.to_csv(os.path.join(split_output_dir, "train.csv"), index=False)

    shutil.copy(valid_split_path, os.path.join(split_output_dir, "valid.csv"))
    shutil.copy(test_split_path, os.path.join(split_output_dir, "test.csv"))
