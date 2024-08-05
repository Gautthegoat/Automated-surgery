import torch

class Config:
    def __init__(self):
        # General settings
        self.name = "ACT_test"
        self.device = torch.device("cuda:1")
        self.type_predictions = "Absolute_joints"  # options: Absolute_joints, Delta_joints
        
        # Data settings
        self.data_dir = "/data/gb/automated_surgery_dataset/"
        self.image_size = (int(1080/3), int(1920/3))  # (height, width)
        self.num_workers = 8
        self.type_norm = "meanstd"  # options: minmax, meanstd, cos_sin or None
        self.stats = []  # This list will store the normalization stats of the data (this is just so we save the values to be able to use the model for inference)
        
        # Model architecture
        self.num_joints = 12 # 12 if you want to use raw values
        self.action_dim = 12  # assuming each action specifies target positions for all 6 joints
        self.chunk_size = 100  # number of timesteps to predict at once / Here 30 fps -> 3.33s
        self.take_current_actions = True  # If True, the model will take the current actions as input
        
        # ResNet settings
        self.resnet_type = 18  # options: 18, 34, 50
        self.pretrained = True
        self.type_Norm = "GroupNorm"  # options: BatchNorm2d, GroupNorm if GropNorm, the model won't use pretrained weights
        self.freeze_BatchNorm = False  # only used if type_Norm is BatchNorm2d
        self.num_GroupNorm = 32  # only used if type_Norm is GroupNorm
        self.layers = [1, 2, 3, 4]  # Layers used as input for the transformer encoder (1-4) -> Not implemented yet
        
        # Transformer settings
        self.embed_dim = 512  # Has to stay 512 since we're using a pre-trained ResNet
        self.num_encoder_layers = 4
        self.num_decoder_layers = 7
        self.num_heads = 8
        self.feedforward_dim = 3200
        self.dropout = 0.1
        
        # CVAE settings
        self.latent_dim = 32
        self.beta = 10  # weight for KL divergence loss
        
        # Training settings
        self.batch_size = 8
        self.num_epochs = 100
        self.learning_rate = 1e-5
        self.weight_decay = 1e-4
        self.use_clip_grad_norm = False
        self.clip_grad_norm = 1.0
        
        # Logging and checkpoints
        self.log_dir = f"Archive/logs/{self.name}"
        self.checkpoint_dir = f"Archive/checkpoints/{self.name}"
        self.save_frequency = 10  # save model every N epochs
            
    def to_dict(self):
        return {key: self._format_value(value) for key, value in vars(self).items() if not key.startswith('__')}

    def _format_value(self, value):
        if isinstance(value, torch.device):
            return str(value)
        elif isinstance(value, tuple):
            return list(value)
        else:
            return value
        
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config

if __name__ == "__main__":
    config = Config()
    print(config.to_dict())