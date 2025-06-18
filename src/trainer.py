import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from .data import MultiModalDataset
from .models import FourModalFusion
from .utils import calculate_weighted_metric, verify_model_alignment

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.load_feature_columns()
        self.setup_logging_and_dirs()

    def load_feature_columns(self):
        # These are fixed for this project, but could also be in the config
        self.cl_columns = ['有无胆结石史', '年龄','A/GRatio', 'ALP', 'PA', 'DBIL', 'TBIL', 
                           'CEA', 'AFP', 'CA19-9', 'CA15-3', 'CA125']
        self.cl_columns_continue = ['年龄','A/GRatio', 'ALP', 'PA', 'DBIL', 'TBIL', 
                                  'CEA', 'AFP', 'CA19-9', 'CA15-3', 'CA125']
        
        try:
            ra_path = self.config['paths']['ra_feature_names_path']
            self.ra_columns = np.loadtxt(ra_path, dtype=str)
        except OSError as e:
            print(f"Error loading ra_columns from {ra_path}: {e}")
            raise

    def setup_logging_and_dirs(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(self.config['output_dir']) / f"experiment_{timestamp}"
        self.dirs = {
            'checkpoints': self.exp_dir / 'checkpoints',
            'logs': self.exp_dir / 'logs',
            'predictions': self.exp_dir / 'predictions',
            'metrics': self.exp_dir / 'metrics',
            'validation': self.exp_dir / 'validation',
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        log_file = self.dirs['logs'] / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        with open(self.dirs['logs'] / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4, default=str)
        self.logger.info("Experiment setup complete. Config and directories saved.")

    def prepare_data(self):
        self.logger.info("Preparing data...")
        df = pd.read_csv(self.config['paths']['csv_path'])
        cl_data = pd.read_excel(self.config['paths']['cl_data_path'])
        ra_data = pd.read_csv(self.config['paths']['ra_data_path'])

        cl_data['ID'] = cl_data['ID'].astype('string')
        ra_data['ID'] = ra_data['ID'].astype('string')
        df['patient_id'] = df['patient_id'].astype('string')
        
        drop_names = ['RN_408', 'RN_436', 'RN_086', 'RN_204', 'RN_281', 
                      'XH0497', 'XH0618', 'XH0535', 'XH0345']
        df = df[~df['patient_id'].isin(drop_names)]
    
        internal_data = df[df['source_type'] == 'internal'].copy()
        
        merged_internal = pd.merge(internal_data, cl_data[['ID'] + list(self.cl_columns)], left_on='patient_id', right_on='ID', how='left')
        merged_internal = pd.merge(merged_internal, ra_data[['ID'] + list(self.ra_columns)], left_on='patient_id', right_on='ID', how='left')
        merged_internal = merged_internal.drop(['ID_x', 'ID_y'], axis=1, errors='ignore')

        self.logger.info(f"Internal data shape: {merged_internal.shape}")

        scaler_cl = StandardScaler().set_output(transform='pandas')
        scaler_cl.fit(merged_internal[self.cl_columns_continue])
        merged_internal[self.cl_columns_continue] = scaler_cl.transform(merged_internal[self.cl_columns_continue])
    
        scaler_ra = StandardScaler().set_output(transform='pandas')
        scaler_ra.fit(merged_internal[self.ra_columns])
        merged_internal[self.ra_columns] = scaler_ra.transform(merged_internal[self.ra_columns])
        
        self.data = merged_internal
        self.logger.info("Data preparation and standardization complete.")

    def run(self):
        self.prepare_data()
        
        self.model_paths = verify_model_alignment(
            self.config['paths']['experiment_base_path'], 
            self.config['paths']['concat_base_path']
        )

        for fold in range(self.config['num_folds']):
            self.train_fold(fold)

    def train_fold(self, fold):
        self.logger.info(f"\\n========== Training Fold {fold} ==========")
        fold_dir = self.dirs['checkpoints'] / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        
        model = FourModalFusion(
            cl_feature_len=len(self.cl_columns),
            ra_feature_len=len(self.ra_columns),
            concat_model_path=self.model_paths['concat']['paths'][fold],
            image_model1_path=self.model_paths['enlarged']['paths'][fold],
            image_model2_path=self.model_paths['original']['paths'][fold],
            reduction_factor=self.config['hparams']['reduction_factor'],
            hidden_dropout=self.config['hparams']['hidden_dropout'],
            final_dropout=self.config['hparams']['final_dropout'],
            device=self.device
        ).to(self.device)

        train_df = self.data[self.data['fold'] != fold].copy()
        val_df = self.data[self.data['fold'] == fold].copy()

        train_dataset = MultiModalDataset(df=train_df, base_path=self.config['paths']['data_path'], cl_cols=self.cl_columns, ra_cols=self.ra_columns, is_train=True)
        val_dataset = MultiModalDataset(df=val_df, base_path=self.config['paths']['data_path'], cl_cols=self.cl_columns, ra_cols=self.ra_columns, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=self.config['hparams']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['hparams']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['hparams']['learning_rate'], weight_decay=self.config['hparams']['weight_decay'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=self.config['hparams']['scheduler_factor'], patience=self.config['hparams']['scheduler_patience'], min_lr=self.config['hparams']['min_lr'])

        best_weighted_metric = -float('inf')
        epochs_no_improve = 0

        for epoch in range(self.config['hparams']['num_epochs']):
            self.logger.info(f"--- Epoch {epoch+1}/{self.config['hparams']['num_epochs']} ---")
            
            train_loss = self.train_epoch(model, train_loader, optimizer)
            val_loss, val_preds, val_labels = self.evaluate_epoch(model, val_loader)

            val_auc = roc_auc_score(val_labels, val_preds)
            val_ap = average_precision_score(val_labels, val_preds)
            current_weighted_metric = calculate_weighted_metric(val_auc, val_ap)

            self.logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | Weighted: {current_weighted_metric:.4f}")

            scheduler.step(current_weighted_metric)

            if current_weighted_metric > best_weighted_metric:
                best_weighted_metric = current_weighted_metric
                epochs_no_improve = 0
                best_model_path = fold_dir / f'best_model_epoch_{epoch}_auc{val_auc:.4f}.pth'
                torch.save(model.state_dict(), best_model_path)
                self.logger.info(f"New best model saved to {best_model_path}")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= self.config['hparams']['early_stopping_patience']:
                self.logger.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
                break

    def train_epoch(self, model, loader, optimizer):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            orig_image, orig_mask, enla_image, enla_mask, cl_features, ra_features, labels, _ = batch
            
            orig_image, orig_mask = orig_image.to(self.device), orig_mask.to(self.device)
            enla_image, enla_mask = enla_image.to(self.device), enla_mask.to(self.device)
            cl_features, ra_features = cl_features.to(self.device), ra_features.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            
            outputs = model(orig_image, orig_mask, enla_image, enla_mask, cl_features, ra_features)
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(loader)

    def evaluate_epoch(self, model, loader):
        model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        pbar = tqdm(loader, desc="Validating")
        with torch.no_grad():
            for batch in pbar:
                orig_image, orig_mask, enla_image, enla_mask, cl_features, ra_features, labels, _ = batch
                
                orig_image, orig_mask = orig_image.to(self.device), orig_mask.to(self.device)
                enla_image, enla_mask = enla_image.to(self.device), enla_mask.to(self.device)
                cl_features, ra_features = cl_features.to(self.device), ra_features.to(self.device)
                labels = labels.to(self.device)

                outputs = model(orig_image, orig_mask, enla_image, enla_mask, cl_features, ra_features)
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels)
                
                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(outputs).squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(loader), np.array(all_preds), np.array(all_labels) 