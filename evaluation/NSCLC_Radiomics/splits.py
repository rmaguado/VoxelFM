import os
import random
import pandas as pd

random.seed(4)

DATA_ROOT = "/scratch/VM/radio-foundation/datasets/NSCLC-Radiomics"
OUTPUT_PATH = "evaluation/NSCLC_Radiomics/splits"

meta_df = pd.read_csv(os.path.join(DATA_ROOT, "metadata.csv"))
meta_df = meta_df[["PatientID", "gender"]]
meta_df.head()

male_uids = meta_df[meta_df["gender"] == "male"]["PatientID"].to_list()
female_uids = meta_df[meta_df["gender"] == "female"]["PatientID"].to_list()
random.shuffle(male_uids)
random.shuffle(female_uids)

n_train_male = int(0.7 * len(male_uids))
n_train_female = int(0.7 * len(female_uids))

n_valid_male = int(0.15 * len(male_uids))
n_valid_female = int(0.15 * len(female_uids))

train_uids = male_uids[:n_train_male] + female_uids[:n_train_female]
valid_uids = (
    male_uids[n_train_male : n_train_male + n_valid_male]
    + female_uids[n_train_female : n_train_female + n_valid_female]
)
test_uids = (
    male_uids[n_train_male + n_valid_male :]
    + female_uids[n_train_female + n_valid_female :]
)

random.shuffle(train_uids)
random.shuffle(valid_uids)
random.shuffle(test_uids)

train_df = pd.DataFrame({"series_uid": train_uids})
valid_df = pd.DataFrame({"series_uid": valid_uids})
test_df = pd.DataFrame({"series_uid": test_uids})

train_df.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
valid_df.to_csv(os.path.join(OUTPUT_PATH, "valid.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), index=False)
