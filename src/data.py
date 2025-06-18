import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity

class MultiModalDataset(Dataset):
    """Dataset for loading image data and clinical/radiomics features"""
    def __init__(self, df, base_path, cl_cols, ra_cols, is_train=True):
        self.df = df
        self.base_path = Path(base_path)
        self.cl_columns = cl_cols
        self.ra_columns = ra_cols
        self.is_train = is_train
        
        self.base_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity()
        ])
        
        # Data augmentation is disabled as in the original script.
        # The code is kept here for future reference.
        self.spatial_transforms = None
        self.intensity_transforms = None
        
        self._validate_and_process_data()
        
    def _validate_and_process_data(self):
        """Validate file existence and data integrity"""
        valid_samples = []
        
        for _, row in self.df.iterrows():
            patient_id = row['patient_id']
            batch_num = row['batch_number']
            case_num = row['case_number']
            
            orig_image = self.base_path / "original_spacing" / batch_num / f"{case_num[:-5]}_s_image.nii.gz"
            orig_mask = self.base_path / "original_spacing" / batch_num / f"{case_num[:-5]}_s_mask.nii.gz"
            enla_image = self.base_path / "enlarged_spacing" / batch_num / f"{case_num[:-5]}_l_image.nii.gz"
            enla_mask = self.base_path / "enlarged_spacing" / batch_num / f"{case_num[:-5]}_l_mask.nii.gz"
            
            # For this dataset, we assume columns are already checked and present from the data prep stage
            if all(f.exists() for f in [orig_image, orig_mask, enla_image, enla_mask]):
                valid_samples.append(row)
    
        self.df = pd.DataFrame(valid_samples).reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No valid samples found after validation!")
        
        # Ensure all feature columns are float32
        self.df[self.cl_columns] = self.df[self.cl_columns].astype(np.float32)
        if self.ra_columns.size > 0:
            self.df[self.ra_columns] = self.df[self.ra_columns].astype(np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patient_id']
        
        # Load image data
        orig_image_path = self.base_path / "original_spacing" / row['batch_number'] / f"{row['case_number'][:-5]}_s_image.nii.gz"
        orig_mask_path = self.base_path / "original_spacing" / row['batch_number'] / f"{row['case_number'][:-5]}_s_mask.nii.gz"
        enla_image_path = self.base_path / "enlarged_spacing" / row['batch_number'] / f"{row['case_number'][:-5]}_l_image.nii.gz"
        enla_mask_path = self.base_path / "enlarged_spacing" / row['batch_number'] / f"{row['case_number'][:-5]}_l_mask.nii.gz"
            
        orig_image = self.base_transforms(str(orig_image_path))
        orig_mask = self.base_transforms(str(orig_mask_path))
        enla_image = self.base_transforms(str(enla_image_path))
        enla_mask = self.base_transforms(str(enla_mask_path))

        # Get clinical and radiomics features
        cl_features = row[self.cl_columns].values.astype(np.float32)
        if self.ra_columns.size > 0:
            ra_features = row[self.ra_columns].values.astype(np.float32)
        else:
            ra_features = np.array([], dtype=np.float32)
            
        # Check for NaN values before converting to tensor
        if np.any(np.isnan(cl_features)) or np.any(np.isnan(ra_features)):
            raise ValueError(f"Found NaN values in features for patient {patient_id}")
            
        cl_features = torch.from_numpy(cl_features)
        ra_features = torch.from_numpy(ra_features)
            
        label = torch.tensor(float(row['label']), dtype=torch.float32)
            
        return orig_image, orig_mask, enla_image, enla_mask, cl_features, ra_features, label, patient_id 