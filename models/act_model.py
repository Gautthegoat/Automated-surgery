import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet, BasicBlock

class ACTModel(nn.Module):
    def __init__(self, config):
        super(ACTModel, self).__init__()
        self.config = config

        # ResNet
        self.resnet = get_resnet(self.config, print_resnet=False)

        # Projection layers
        self.joint_proj = nn.Linear(config.num_joints, config.embed_dim)
        self.z_proj = nn.Linear(config.latent_dim, config.embed_dim)
        
        # Get size of output from ResNet
        self.size_output_resnet = self.resnet(torch.randn(1, 3, config.image_size[0], config.image_size[1]))
        self.size_output_resnet = self.size_output_resnet.flatten(2).shape

        # Positional encoding for the encoder
        self.encoder_input_embed = nn.Embedding(self.size_output_resnet[2]+2, config.embed_dim)  # (H*W/32^2+2, embed_dim)
        # Query position embedding for decoder
        self.query_embed = nn.Embedding(config.chunk_size, config.embed_dim)  # (chunk_size, embed_dim)
        
        # Style encoder
        self.style_encoder = StyleEncoder(config.num_joints, config.embed_dim, config.latent_dim, config)
        
        # Main transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True
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
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(config.embed_dim, config.action_dim)
        
    def forward(self, image, current_joints, future_joints=None, training=False):
        """
        current_joints: (batch_size, 1, num_joints)
        future_joints: (batch_size, chunk_size, num_joints)
        """
        img_features = self.resnet(image)  # (batch_size, embed_dim, H/32, W/32)
        img_features = img_features.flatten(2).transpose(1, 2)  # (batch_size, H*W/32^2, embed_dim)
        
        # Process current joints
        current_joint_features = self.joint_proj(current_joints)  # (batch_size, 1, embed_dim)
        
        # Generate style variable z
        if training:
            z, mu, logvar = self.style_encoder(current_joints, future_joints)  # z: (batch_size, latent_dim)
        else:
            z = torch.zeros(image.size(0), self.config.latent_dim).to(image.device)  # (batch_size, latent_dim)
            mu, logvar = None, None
        
        z_projected = self.z_proj(z).unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Combine features for transformer encoder
        encoder_input = torch.cat([img_features, current_joint_features, z_projected], dim=1)  # (batch_size, H*W/32^2+2, embed_dim)

        # Add positional encoding
        encoder_positional_encoding = self.encoder_input_embed.weight.unsqueeze(0).repeat(image.size(0), 1, 1)  # (batch_size, H*W/32^2+2, embed_dim)
        encoder_input = encoder_input + encoder_positional_encoding

        # Transformer encoder
        memory = self.transformer_encoder(encoder_input)  # (batch_size, H*W/32^2+2, embed_dim)

        # Prepare decoder input
        tgt = self.query_embed.weight.unsqueeze(0).repeat(image.size(0), 1, 1)  # (batch_size, chunk_size, embed_dim)

        # Transformer decoder
        output = self.transformer_decoder(tgt, memory)  # (batch_size, chunk_size, embed_dim)
        
        # Generate action sequence
        action_sequence = self.output_layer(output)  # (batch_size, chunk_size, action_dim)

        return action_sequence, mu, logvar  # (batch_size, chunk_size, action_dim), (batch_size, latent_dim), (batch_size, latent_dim)
    
class StyleEncoder(nn.Module):
    def __init__(self, joint_dim, embed_dim, latent_dim, config):
        super(StyleEncoder, self).__init__()

        # Project joints to embedding space
        self.joint_proj = nn.Linear(joint_dim, embed_dim)

        # CLS token # TODO Decide if we want to dropout(0.1) here... could be useful
        self.cls_token = nn.Embedding(1, embed_dim)  # (1, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def positional_encoding(self, x):
        batch_size, seq_len, d_model = x.size()
        
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        return x + pe.to(x.device)
        
    def forward(self, current_joints, future_joints):
        # Project joints to embedding space
        current_embed = self.joint_proj(current_joints)  # (batch_size, 1, embed_dim)
        future_embed = self.joint_proj(future_joints)  # (batch_size, chunk_size, embed_dim)
        
        # Create CLS token
        cls_token = self.cls_token.weight.unsqueeze(0).repeat(current_embed.size(0), 1, 1)  # (batch_size, 1, embed_dim)
        
        # Combine for transformer input
        transformer_input = torch.cat([cls_token, current_embed, future_embed], dim=1)  # (batch_size, chunk_size+2, embed_dim)

        # Add positional encoding
        transformer_input = self.positional_encoding(transformer_input)
        
        # Pass through transformer
        output = self.transformer_encoder(transformer_input)  # (batch_size, chunk_size+2, embed_dim)
        
        # Use the CLS token output for mu and logvar
        cls_output = output[:, 0, :]  # (batch_size, embed_dim)

        mu = self.fc_mu(cls_output)  # (batch_size, latent_dim)
        logvar = self.fc_logvar(cls_output)  # (batch_size, latent_dim)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
class ResNetOutputAdapter(nn.Module):
    def __init__(self, input_channels, target_channels):
        super(ResNetOutputAdapter, self).__init__()
        self.adapters = nn.ModuleDict()
        for layer_name, channels in input_channels.items():
            self.adapters[layer_name] = nn.Conv2d(channels, target_channels, kernel_size=1)
        
    def forward(self, x):
        return {layer_name: self.adapters[layer_name](tensor) 
                for layer_name, tensor in x.items()}
    
class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
    
    def forward(self, x):
        scale = self.weight * (self.running_var + 1e-5).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
    
def adapt_resnet_output(resnet_output, target_channels=512):
    # Get the number of channels in the ResNet output for each layer
    input_channels = {layer_name: tensor.shape[1] for layer_name, tensor in resnet_output.items()}

    # Create the adapter
    adapter = ResNetOutputAdapter(input_channels, target_channels)

    # Apply the adapter to the ResNet output
    adapted_output = adapter(resnet_output)
    print(adapted_output)

    return adapted_output
    
def get_resnet(config, print_resnet=False):
    """
    Obtain a ResNet18 model with the specified settings
    """
    # Modify normalization layers based on config
    if config.type_Norm == "BatchNorm2d":
        norm_layer = lambda x: FrozenBatchNorm2d(x) if config.freeze_BatchNorm else nn.BatchNorm2d(x)
    elif config.type_Norm == "GroupNorm":
        norm_layer = lambda x: nn.GroupNorm(config.num_GroupNorm, x)

    resnet = ResNet(BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer)

    if config.pretrained and config.type_Norm == "BatchNorm2d":
        state_dict = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
        resnet.load_state_dict(state_dict, strict=False)

    resnet = nn.Sequential(
        *list(resnet.children())[:-2] 
    )

    if print_resnet:
        print(resnet)

    return resnet

def loss_function(pred_actions, true_actions, mu, logvar, config):
    reconstruction_loss = F.l1_loss(pred_actions[0, 0, :], true_actions[0, 0, :])
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + config.beta * kl_divergence