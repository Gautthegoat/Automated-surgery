import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from utils.config import config

class ACTModel(nn.Module):
    def __init__(self):
        super(ACTModel, self).__init__()
        
        # Image encoder (ResNet)
        self.resnet = resnet18(pretrained=config.pretrained)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Projection layers
        self.joint_proj = nn.Linear(config.num_joints, config.embed_dim)
        self.z_proj = nn.Linear(config.latent_dim, config.embed_dim)

        # Query embedding
        self.query_embed = nn.Embedding(config.chunk_size, config.embed_dim)  # (chunk_size, embed_dim)
        
        # Style encoder
        self.style_encoder = StyleEncoder(config.num_joints, config.embed_dim, config.latent_dim)
        
        # Main transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(config.embed_dim, config.action_dim)
        
    def forward(self, image, current_joints, future_joints=None, training=False):
        """
        current_joints: (batch_size, num_joints)
        future_joints: (batch_size, chunk_size, num_joints)
        """
        # Process image
        img_features = self.resnet(image)  # (batch_size, embed_dim, H/32, W/32)
        img_features = img_features.flatten(2)  # (batch_size, embed_dim, H*W/32^2)
        img_features = img_features.permute(2, 0, 1)  # (H*W/32^2, batch_size, embed_dim)
        
        # Process current joints
        current_joint_features = self.joint_proj(current_joints).unsqueeze(0)  # (1, batch_size, embed_dim)
        
        # Generate style variable z
        if training:
            z, mu, logvar = self.style_encoder(current_joints, future_joints)  # z: (batch_size, embed_dim)
        else:
            z = torch.zeros(image.size(0), config.latent_dim).to(image.device)  # (batch_size, latent_dim)
            mu, logvar = None, None
        
        z_projected = self.z_proj(z).unsqueeze(0)  # (1, batch_size, embed_dim)
        
        # Combine features for transformer encoder
        encoder_input = torch.cat([img_features, current_joint_features, z_projected], dim=0)  # (H*W/32^2+2, batch_size, embed_dim)
        
        # Add positional encoding
        encoder_input = self.positional_encoding(encoder_input)
        
        # Transformer encoder
        memory = self.transformer_encoder(encoder_input)  # (H*W/32^2+2, batch_size, embed_dim)
        
        # Prepare decoder input
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, image.size(0), 1)  # (chunk_size, batch_size, embed_dim)
        tgt = self.positional_encoding(tgt)  # (chunk_size, batch_size, embed_dim)

        # Transformer decoder
        output = self.transformer_decoder(tgt, memory)  # (chunk_size, batch_size, embed_dim)
        
        # Generate action sequence
        action_sequence = self.output_layer(output)
        
        return action_sequence.transpose(0, 1), mu, logvar  # (batch_size, chunk_size, action_dim), (batch_size, latent_dim), (batch_size, latent_dim)
    
    def positional_encoding(self, x):
        seq_len, batch_size, d_model = x.size()
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1).repeat(1, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position[:, 0::2] * div_term)
        pe[:, 0, 1::2] = torch.cos(position[:, 1::2] * div_term)
        return x + pe.to(x.device)

class StyleEncoder(nn.Module):
    def __init__(self, joint_dim, embed_dim, latent_dim):
        super(StyleEncoder, self).__init__()
        self.joint_proj = nn.Linear(joint_dim, embed_dim)
        self.cls_token = nn.Embedding(1, embed_dim)  # (1, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
    def forward(self, current_joints, future_joints):
        # Project joints to embedding space
        current_embed = self.joint_proj(current_joints).unsqueeze(0)  # (1, batch_size, embed_dim)
        future_embed = self.joint_proj(future_joints).transpose(0, 1)  # (chunk_size, batch_size, embed_dim)
        
        # Create CLS token
        cls_token = self.cls_token.weight.unsqueeze(0) # (1, 1, embed_dim)
        cls_token = cls_token.repeat(1, current_embed.size(1), 1)  # (1, batch_size, embed_dim)
        
        # Combine for transformer input
        transformer_input = torch.cat([cls_token, current_embed, future_embed], dim=0) # (chunk_size+2, batch_size, embed_dim)
        
        # Pass through transformer
        output = self.transformer_encoder(transformer_input)
        
        # Use the CLS token output for mu and logvar
        cls_output = output[0]
        
        mu = self.fc_mu(cls_output)
        logvar = self.fc_logvar(cls_output)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar

def loss_function(pred_actions, true_actions, mu, logvar):
    reconstruction_loss = F.mse_loss(pred_actions, true_actions, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + config.beta * kl_divergence