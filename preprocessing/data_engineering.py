import pandas as pd
from sklearn.utils import shuffle
import boto3
from io import StringIO

# AWS setup
bucket = "diabetes-directory"
prefix = "02_engineered"
output_files = {
    "full":  "prepared_diabetes_full.csv",
    "train": "prepared_diabetes_train.csv",
    "test":  "prepared_diabetes_test.csv"
}

s3_client = boto3.client("s3")

def CSV_Reader():
    return pd.read_csv(f"s3://{bucket}/01_raw/Diabetes_Input.csv", low_memory=False)

def upload_df_to_s3(df: pd.DataFrame, key: str):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    response = s3_client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{key}",
        Body=csv_buffer.getvalue()
    )
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 200:
        print(f"✅ Uploaded `{key}`: {len(df)} rows to s3://{bucket}/{prefix}/{key}")
    else:
        print(f"❗Failed uploading `{key}`, status code {status}")

# Load and process data
diabetes = CSV_Reader()

# Drop and reorganize columns
reorganized_columns = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient","number_emergency","number_inpatient","number_diagnoses",
    "repaglinide","nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", 
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", 
    "tolazamide", "examide", "citoglipton", "glyburide.metformin", "glipizide.metformin", 
    "glimepiride.pioglitazone", "metformin.rosiglitazone", "metformin.pioglitazone",
    "encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty",
    "readmitted", "metformin", "glipizide", "glyburide", "insulin", "change", "diabetesMed",
    "diag_1", "diag_2", "diag_3", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "race", "gender", "A1Cresult", "max_glu_serum"
]
diabetes = diabetes.reindex(columns=reorganized_columns)

# Drop low-variation medication columns
imbalance_cols = reorganized_columns[8:27]
diabetes.drop(columns=imbalance_cols, inplace=True)

# Rename medication columns
Dcolumns = list(diabetes.columns)
for i in range(14, 18):
    Dcolumns[i] = "medication_" + Dcolumns[i]
Dcolumns[19] = "any_medication"
diabetes.columns = Dcolumns

# Drop unneeded columns
diabetes.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"], inplace=True)

# Recode readmitted column
diabetes['readmitted'] = diabetes['readmitted'].replace(['NO', '>30', '<30'], [0, 0, 1])
diabetes = pd.concat([diabetes["readmitted"], diabetes.drop(["readmitted"], axis=1)], axis=1)

# Convert medication indicators to binary
for med in ["medication_metformin", "medication_glipizide", "medication_glyburide", "medication_insulin"]:
    diabetes[med] = diabetes[med].replace(['No', 'Steady', 'Down', 'Up'], [0, 1, 1, 1])
diabetes['change'] = diabetes['change'].replace(['No', 'Ch'], [0, 1])
diabetes['any_medication'] = diabetes['any_medication'].replace(['Yes', 'No'], [1, 0])

# Map diagnosis codes
def Convert_Disease_Codes(min_code, max_code, newname):
    for col in ['diag_1', 'diag_2', 'diag_3']:
        diabetes[col] = diabetes[col].apply(lambda x: newname if pd.to_numeric(x, errors='coerce') in range(min_code, max_code) else x)

disease_ranges = [
    (340, 459, 'circulatory'), (785, 786, 'circulatory'), (745, 748, 'circulatory'), (459, 460, 'circulatory'),
    (460, 520, 'respiratory'), (786, 787, 'respiratory'), (748, 749, 'respiratory'),
    (520, 580, 'digestive'), (787, 788, 'digestive'), (749, 752, 'digestive'),
    (800, 1000, 'injury'), (710, 740, 'musculoskeletal'), (754, 757, 'musculoskeletal'),
    (580, 630, 'urogenital'), (788, 789, 'urogenital'), (752, 754, 'urogenital'),
    (140, 240, 'neoplasm'), (1, 140, 'infection'),
    (290, 320, 'mentaldis'), (280, 290, 'blooddis'), (320, 360, 'nervous'), (360, 390, 'nervous'),
    (740, 743, 'nervous'), (630, 680, 'pregnancy'),
    (780, 782, 'other'), (784, 785, 'other'), (790, 800, 'other'), (743, 745, 'other'), (757, 760, 'other'),
    (240, 250, 'metabolic'), (251, 280, 'metabolic'),
    (680, 710, 'skin'), (782, 783, 'skin'), (789, 790, 'other'), (783, 784, 'metabolic')
]
for r in disease_ranges:
    Convert_Disease_Codes(*r)

# Replace V/E codes with "injury"
for col in ['diag_1', 'diag_2', 'diag_3']:
    diabetes[col] = diabetes[col].replace(r'^[VE]\d+', 'injury', regex=True)

# Replace remaining numeric diagnosis with 'Nothing'
for col in ['diag_1', 'diag_2', 'diag_3']:
    diabetes[col] = diabetes[col].apply(lambda x: 'Nothing' if pd.to_numeric(x, errors='coerce') is not None else x)

# Recode and clean other columns
diabetes['age'] = diabetes['age'].replace({'[0-10)':1, '[10-20)':2, '[20-30)':3, '[30-40)':4, '[40-50)':5, 
                                           '[50-60)':6, '[60-70)':7, '[70-80)':8, '[80-90)':9, '[90-100)':10})
diabetes['admission_type_id'] = diabetes['admission_type_id'].replace({8:6, 6:5})
discharge_map = dict(zip(range(1,31), ['home', 'hospital', 'nursing', 'nursing', 'hospice', 'hhealth', 'leftAMA', 
                                       'hhealth', 'hospital', 'hospital','died', 'hospital', 'hospice', 'hospice', 
                                       'hospital', 'outpatient', 'outpatient', 'unknown','died', 'died', 'died', 
                                       'outpatient', 'hospital', 'nursing', 'unknown', 'unknown', 'nursing', 
                                       'psych','hospital', 'outpatient']))
diabetes['discharge_disposition_id'] = diabetes['discharge_disposition_id'].replace(discharge_map)
diabetes.rename(columns={"discharge_disposition_id": "discharge_disposition"}, inplace=True)
diabetes = diabetes[diabetes["discharge_disposition"] != "died"]
diabetes['admission_source_id'] = diabetes['admission_source_id'].clip(upper=8)
diabetes['race'] = diabetes['race'].fillna("Other")
diabetes['gender'] = diabetes['gender'].replace(['Unknown/Invalid'], ['Female'])
diabetes['A1Cresult'] = diabetes['A1Cresult'].replace(['None'], ['NotTaken'])
diabetes['max_glu_serum'] = diabetes['max_glu_serum'].replace(['None'], ['NotTaken'])

# Create diagnosis dummies
def Create_Combined_Diagnosis_Dummies(df, cols):
    all_vals = pd.concat([df[c] for c in cols])
    unique_vals = all_vals.unique()
    dummies = pd.DataFrame(0, index=df.index, columns=[f"diagnosis_{val}" for val in unique_vals])
    for val in unique_vals:
        for c in cols:
            dummies[f"diagnosis_{val}"] |= (df[c] == val).astype(int)
    return dummies

diagnosis_dummies = Create_Combined_Diagnosis_Dummies(diabetes, ['diag_1', 'diag_2', 'diag_3'])
diabetes = pd.concat([diabetes.drop(['diag_1', 'diag_2', 'diag_3'], axis=1), diagnosis_dummies], axis=1)

# Dummify remaining categoricals
def Replace_With_Dummies(df, dummy_cols):
    df2 = df.copy()
    for col in dummy_cols:
        top = df2[col].value_counts().idxmax()
        dummies = pd.get_dummies(df2[col], prefix=col).drop(f"{col}_{top}", axis=1)
        df2 = pd.concat([df2.drop(col, axis=1), dummies], axis=1)
    return df2

dummy_vars = ['race', 'age', 'gender', 'discharge_disposition', 'max_glu_serum', 
              'A1Cresult', 'admission_type_id', 'admission_source_id']
diabetes = Replace_With_Dummies(diabetes, dummy_vars)

# Ensure all values are int
for col in diabetes.columns:
    diabetes[col] = diabetes[col].astype(int)


# Shuffle and split
diabetes = shuffle(diabetes, random_state=42).reset_index(drop=True)
split_idx = int(len(diabetes) * 0.8)
train = diabetes.iloc[:split_idx]
test = diabetes.iloc[split_idx:]

# Upload datasets
upload_df_to_s3(diabetes, output_files["full"])
upload_df_to_s3(train,    output_files["train"])
upload_df_to_s3(test,     output_files["test"])

print("\n✅ All exports complete:")
for label in ["full", "train", "test"]:
    print(f" • {label.capitalize()}: {output_files[label]} ({prefix}/{output_files[label]})")