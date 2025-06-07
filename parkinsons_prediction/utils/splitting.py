import numpy as np

def create_patient_splits(data, test_size=0.2, val_size=0.2, random_state=42):
    """Split data by patients to avoid data leakage."""
    unique_patients = data['patient_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_patients)
    
    n_patients = len(unique_patients)
    test_end = int(n_patients * test_size)
    val_end = test_end + int(n_patients * val_size)
    
    test_patients = set(unique_patients[:test_end])
    val_patients = set(unique_patients[test_end:val_end])
    train_patients = set(unique_patients[val_end:])
    
    train_data = data[data['patient_id'].isin(train_patients)].copy()
    val_data = data[data['patient_id'].isin(val_patients)].copy()
    test_data = data[data['patient_id'].isin(test_patients)].copy()
    
    return train_data, val_data, test_data