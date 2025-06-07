import pandas as pd
from parkinsons_prediction.data.loader import load_data

class ParkinsonsDataPreprocessor:
    """Data preprocessor for Parkinsons disease progression prediction."""
    
    def __init__(self, target_horizons, 
                 test_visit_months,
                 updrs_parts,
                 max_updrs_score):
        
        self.target_horizons = target_horizons
        self.test_visit_months = test_visit_months
        self.updrs_parts = updrs_parts
        self.max_updrs_score = max_updrs_score

    def _add_engineered_features(self, base_frame, protein_data, clinical_data):
        """Add engineered features to the base dataframe."""
        base_frame = base_frame.copy()

        # Visit availability features
        for month in self.test_visit_months:
            clinical_patients = set(clinical_data[clinical_data["visit_month"] == month]["patient_id"])
            base_frame[f"visit_{month}m"] = base_frame.apply(
                lambda row: int((row["patient_id"] in clinical_patients) and (row["visit_month"] >= month)), axis=1
            )

            protein_patients = set(protein_data[protein_data["visit_month"] == month]["patient_id"])
            base_frame[f"btest_{month}m"] = base_frame.apply(
                lambda row: int((row["patient_id"] in protein_patients) and (row["visit_month"] >= month)), axis=1
            )

            # Time-based features
            base_frame[f"t_month_eq_{month}"] = (base_frame["target_month"] == month).astype(int)
            base_frame[f"v_month_eq_{month}"] = (base_frame["visit_month"] == month).astype(int)

        for horizon in self.target_horizons:
            base_frame[f"hor_eq_{horizon}"] = (base_frame["horizon"] == horizon).astype(int)

        base_frame["horizon_scaled"] = base_frame["horizon"] / max(self.target_horizons)

        protein_visit_ids = set(protein_data["visit_id"].unique())
        base_frame["blood_taken"] = base_frame["visit_id"].apply(
            lambda visit_id: int(visit_id in protein_visit_ids)
        )

        visit_history = clinical_data.groupby("patient_id")["visit_month"].apply(set).to_dict()
        base_frame["count_non12_visits"] = base_frame.apply(
            lambda row: len([month for month in visit_history.get(row["patient_id"], [])
                           if month <= row["visit_month"] and month % 12 != 0]), axis=1
        )

        return base_frame

    def transform_train_data(self, protein_data, clinical_data):
        """Transform training data for model input."""
        merged_data = clinical_data.rename(columns={
            "visit_month": "target_month",
            "visit_id": "visit_id_target"
        }).merge(
            clinical_data[["patient_id", "visit_month", "visit_id"]],
            on="patient_id",
            how="left"
        )

        merged_data["horizon"] = merged_data["target_month"] - merged_data["visit_month"]
        filtered_data = merged_data[
            merged_data["horizon"].isin(self.target_horizons) &
            merged_data["visit_month"].isin(self.test_visit_months)
        ]

        enriched_data = self._add_engineered_features(
            base_frame=filtered_data,
            protein_data=protein_data[protein_data["visit_month"].isin(self.test_visit_months)],
            clinical_data=clinical_data[clinical_data["visit_month"].isin(self.test_visit_months)]
        )

        records = []
        for part_number in self.updrs_parts:
            if f"updrs_{part_number}" not in enriched_data.columns:
                continue

            part_data = enriched_data.copy()
            part_data["target"] = part_data[f"updrs_{part_number}"]
            part_data["target_norm"] = part_data["target"] / self.max_updrs_score
            part_data["target_i"] = part_number
            records.append(part_data)

        full_dataset = pd.concat(records, axis=0).reset_index(drop=True)
        
        updrs_cols = [f"updrs_{i}" for i in self.updrs_parts if f"updrs_{i}" in full_dataset.columns]
        full_dataset.drop(columns=updrs_cols, inplace=True)

        for part_number in self.updrs_parts:
            full_dataset[f"target_n_{part_number}"] = (full_dataset["target_i"] == part_number).astype(int)

        return full_dataset

    def prepare_training_data(self, protein_data, clinical_data, supplemental_data):
        """Prepare both training and supplemental data."""
        training_sample = self.transform_train_data(protein_data, clinical_data)
        training_sample = training_sample[~training_sample["target"].isnull()].copy()
        training_sample["is_suppl"] = 0

        supplemental_sample = self.transform_train_data(protein_data, supplemental_data)
        supplemental_sample = supplemental_sample[~supplemental_sample["target"].isnull()].copy()
        supplemental_sample["is_suppl"] = 1

        return training_sample, supplemental_sample
    
    def transform_test_data(self, protein_data, clinical_data, test_df):
        test_df = test_df.copy()

        test_expanded = []
        for horizon in self.target_horizons:
            for part_number in self.updrs_parts:
                df = test_df.copy()
                df["horizon"] = horizon
                df["target_month"] = df["visit_month"] + horizon
                df["target_i"] = part_number
                test_expanded.append(df)

        test_df = pd.concat(test_expanded, ignore_index=True)

        test_df["is_suppl"] = 0

        test_df = self._add_engineered_features(
            base_frame=test_df,
            protein_data=protein_data,
            clinical_data=clinical_data
        )

        for part_number in self.updrs_parts:
            test_df[f"target_n_{part_number}"] = (test_df["target_i"] == part_number).astype(int)

        return test_df



    

def preprocess_data(cfg: dict) -> pd.DataFrame:
    data_tuple = load_data(cfg)
    mode = cfg.task["mode"]
    preprocessor = ParkinsonsDataPreprocessor(cfg.data['target_horizons'], cfg.data['test_visit_months'],
                                              cfg.data['updrs_parts'], cfg.data['max_updrs_score'])

    if mode == "train":
        proteins, clinical, supplement = data_tuple
        supplement.loc[supplement["visit_month"] == 5, "visit_month"] = 6
        train_df, suppl_df = preprocessor.prepare_training_data(proteins, clinical, supplement)
        data = pd.concat([train_df, suppl_df], axis=0).reset_index(drop=True)

    elif mode == "inference":
        test_proteins, test_clinical = data_tuple
        test_clinical.loc[test_clinical["visit_month"] == 5, "visit_month"] = 6

        data = preprocessor.transform_test_data(
            protein_data=test_proteins[test_proteins["visit_month"].isin(cfg.data['test_visit_months'])],
            clinical_data=test_clinical[test_clinical["visit_month"].isin(cfg.data['test_visit_months'])],
            test_df=test_clinical
        )

    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    data.to_csv(cfg.data[f'preprocessed_{mode}_data_path'], index=False)
    return data
