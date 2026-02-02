import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class CXRDataset(Dataset):
    def __init__(self, dataframe, root_dir, transforms=None, classes=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns.
            root_dir (str): Directory with all the images.
            transforms (albumentations.Compose): Transforms to apply.
            classes (list): List of class names to target.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transforms = transforms
        self.classes = classes
        
        # 1. Identify Image ID/Path Column
        # User schema has 'cxr_filename'
        if 'cxr_filename' in self.dataframe.columns:
            self.image_ids = self.dataframe['cxr_filename'].values
        elif 'cal_filename' in self.dataframe.columns: # typo handling
             self.image_ids = self.dataframe['cal_filename'].values
        elif 'Image Index' in self.dataframe.columns:
            self.image_ids = self.dataframe['Image Index'].values
        else:
            # Fallback: use first column
            print("Warning: Could not find 'cxr_filename' or 'Image Index'. Using first column as image ID.")
            self.image_ids = self.dataframe.iloc[:, 0].values

        # 2. Prepare Labels
        if self.classes:
            # check if all classes exist
            missing = [c for c in self.classes if c not in self.dataframe.columns]
            if missing:
                raise ValueError(f"The following target classes are missing from CSV: {missing}")
            self.labels = self.dataframe[self.classes].values
        else:
            # If no classes provided, just return zeros or try to infer?
            # Safe default
            print("Warning: No classes provided. Returning dummy labels.")
            self.labels = np.zeros((len(self.dataframe), 1))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = str(self.image_ids[idx])
        
        # Robust path construction
        # 1. Try exact match in root_dir
        img_path = os.path.join(self.root_dir, img_id)
        
        # 2. If valid extension missing, try adding common extensions
        if not os.path.exists(img_path):
            common_exts = ['.png', '.jpg', '.jpeg']
            found = False
            for ext in common_exts:
                test_path = img_path + ext
                if os.path.exists(test_path):
                    img_path = test_path
                    found = True
                    break
            
            # 3. Check if it's already a relative path in the CSV (e.g., ./cxrs/file.png)
            if not found:
                # sometimes CSV has full relative path like ./cxrs/img.png
                # and root_dir is just the base
                basename = os.path.basename(img_id)
                img_path = os.path.join(self.root_dir, basename)
                
                # try extensions on basename
                if not os.path.exists(img_path):
                    for ext in common_exts:
                        test_path = img_path + ext
                        if os.path.exists(test_path):
                            img_path = test_path
                            found = True
                            break
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            # Skip or error? Error is safer for debugging setup
            raise FileNotFoundError(f"Image not found at {img_path} (ID: {img_id})")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        # Get label
        label = self.labels[idx]
        
        # Return tensors
        return image, torch.tensor(label, dtype=torch.float32)
