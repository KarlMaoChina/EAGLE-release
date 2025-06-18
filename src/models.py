import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet

# ======================================================================================
# Base Image Model (from gallbladder_classifier_configable_attentionen_dual_optim.py)
# ======================================================================================

class SpatialAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels + 1, in_channels // 4, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(in_channels // 4, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x, mask):
        # Ensure mask size matches the feature map
        mask = torch.nn.functional.interpolate(mask, size=x.shape[2:])
        # Concatenate features and mask
        attention_input = torch.cat([x, mask], dim=1)
        # Generate attention map
        attention_map = self.attention(attention_input)
        # Apply attention
        return x * attention_map

class GallbladderClassifier(torch.nn.Module):
    """Gallbladder Classifier with Spatial Attention"""
    def __init__(self, feature_scale=1.0):
        super().__init__()
        # Base ResNet
        self.model = ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[int(64 * feature_scale), int(128 * feature_scale), 
                          int(256 * feature_scale), int(512 * feature_scale)],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1
        )
        
        # Attention modules for key layers
        self.attention1 = SpatialAttention(int(64 * feature_scale))
        self.attention2 = SpatialAttention(int(128 * feature_scale))
        self.attention3 = SpatialAttention(int(256 * feature_scale))
        
        # ResNet layers
        self.layer0 = torch.nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu)
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool
        self.fc = self.model.fc

    def forward(self, x, mask):
        # Forward pass with attention applied after key layers
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.attention1(x, mask)
        
        x = self.layer2(x)
        x = self.attention2(x, mask)
        
        x = self.layer3(x)
        x = self.attention3(x, mask)
        
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# ======================================================================================
# Fusion Model Components (from train_4modal_v3.py)
# ======================================================================================

class NN_model(nn.Module):
    def __init__(self, in_channels=1, ra_num=0):
        super(NN_model, self).__init__()
        self.in_channels = in_channels
        self.fc0 = nn.Linear(ra_num, 128)
        self.fc = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, inputs_ra):
        x = self.fc0(inputs_ra)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    def get_intermediate_features(self, inputs_ra):
        """Get intermediate layer features for fusion"""
        x = self.fc0(inputs_ra)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc(x)  # Returns 256-dim features
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.residual_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        return x + self.residual_weight * self.block(x)

class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=4)
        
    def forward(self, features_list):
        features_reshaped = [f.unsqueeze(0) for f in features_list]
        
        attended_features = []
        for i, query in enumerate(features_reshaped):
            keys = torch.cat([f for j, f in enumerate(features_reshaped) if j != i], dim=0)
            attended, _ = self.attention(query, keys, keys)
            attended_features.append(attended.squeeze(0))
        
        return torch.cat(attended_features, dim=1)

class FeatureWeighting(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        self.feature_dims = feature_dims
        self.modal_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, dim)) for dim in feature_dims
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid()) for dim in feature_dims
        ])
        
    def forward(self, features_list):
        weighted_features = []
        for features, weights, gate in zip(features_list, self.modal_weights, self.gates):
            gate_value = gate(features)
            weighted = features * weights * gate_value
            weighted = F.layer_norm(weighted, (features.shape[-1],))
            weighted_features.append(weighted)
        return weighted_features

class FourModalFusion(nn.Module):
    def __init__(self, cl_feature_len, ra_feature_len, concat_model_path, image_model1_path, image_model2_path,
                 reduction_factor=2, hidden_dropout=0.3, final_dropout=0.2, device='cuda'):
        super().__init__()
        
        self.concat_model = NN_model(ra_num=ra_feature_len + cl_feature_len).to(device)
        concat_state_dict = torch.load(concat_model_path, map_location=device)
        if isinstance(concat_state_dict, NN_model):
            concat_state_dict = concat_state_dict.state_dict()
        self.concat_model.load_state_dict(concat_state_dict)
        for param in self.concat_model.parameters():
            param.requires_grad = False

        self.image_model1 = GallbladderClassifier(feature_scale=0.4).to(device)
        self.image_model2 = GallbladderClassifier(feature_scale=0.4).to(device)
        self.image_model1.load_state_dict(torch.load(image_model1_path, map_location=device))
        self.image_model2.load_state_dict(torch.load(image_model2_path, map_location=device))
        for model in [self.image_model1, self.image_model2]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Get dimensions
        dummy_img = torch.randn(1, 1, 96, 112, 80).to(device)
        dummy_mask = torch.ones(1, 1, 96, 112, 80).to(device)
        with torch.no_grad():
            img_features = self.extract_image_features(self.image_model1, dummy_img, dummy_mask)
            img_feat_dim = img_features.shape[-1]
        
        self.concat_feat_dim = self.concat_model.fc.out_features

        self.img1_projection = self._create_projection(img_feat_dim, self.concat_feat_dim, hidden_dropout, device)
        self.img2_projection = self._create_projection(img_feat_dim, self.concat_feat_dim, hidden_dropout, device)
        
        self.cross_modal_attention = CrossModalAttention(self.concat_feat_dim).to(device)
        self.feature_weighting = FeatureWeighting([self.concat_feat_dim] * 3).to(device)
        
        total_feat_dim = self.concat_feat_dim * 3
        reduced_dim = total_feat_dim // reduction_factor
        
        self.feat_norm = nn.LayerNorm(total_feat_dim).to(device)
        self.dim_reduction = self._create_projection(total_feat_dim, reduced_dim, hidden_dropout, device)
        
        self.fusion_mlp = nn.Sequential(
            ResidualBlock(reduced_dim, dropout=hidden_dropout),
            ResidualBlock(reduced_dim, dropout=hidden_dropout),
            nn.LayerNorm(reduced_dim),
            nn.Dropout(final_dropout),
            nn.Linear(reduced_dim, 1)
        ).to(device)

    def _create_projection(self, in_dim, out_dim, dropout, device):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ).to(device)

    def extract_image_features(self, model, x, mask):
        x = model.layer0(x)
        x = model.layer1(x)
        x = model.attention1(x, mask)
        x = model.layer2(x)
        x = model.attention2(x, mask)
        x = model.layer3(x)
        x = model.attention3(x, mask)
        x = model.layer4(x)
        x = model.avgpool(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
    
    def forward(self, x_orig, mask_orig, x_enla, mask_enla, cl_features, ra_features):
        with torch.no_grad():
            concat_input = torch.cat([cl_features, ra_features], dim=1)
            concat_features = self.concat_model.get_intermediate_features(concat_input)
            img1_features = self.extract_image_features(self.image_model1, x_enla, mask_enla)
            img2_features = self.extract_image_features(self.image_model2, x_orig, mask_orig)

        img1_proj_features = self.img1_projection(img1_features)
        img2_proj_features = self.img2_projection(img2_features)
        
        features_list = [concat_features, img1_proj_features, img2_proj_features]
        
        attended_features_cat = self.cross_modal_attention(features_list)
        
        # Split features after attention
        attended_features_list = torch.split(attended_features_cat, self.concat_feat_dim, dim=1)
        
        weighted_features = self.feature_weighting(attended_features_list)
        
        combined_features = torch.cat(weighted_features, dim=1)
        combined_features = self.feat_norm(combined_features)
        reduced_features = self.dim_reduction(combined_features)
        out = self.fusion_mlp(reduced_features)
        return out