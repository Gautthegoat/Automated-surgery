import torch

class Config:
    # General settings
    seed = 42
    device = torch.device("cuda:2")
    
    # Data settings
    data_dir = "data/collected_demonstrations"
    num_demonstrations = 50
    demonstration_length = 500  # number of timesteps per demonstration
    image_size = (480, 640)  # (height, width)
    
    # Model architecture
    num_joints = 12
    action_dim = 12  # assuming each action specifies target positions for all 6 joints
    chunk_size = 100  # number of timesteps to predict at once
    
    # ResNet settings
    resnet_type = 18  # options: 18, 34, 50
    pretrained = True
    
    # Transformer settings
    embed_dim = 512
    num_encoder_layers = 4
    num_decoder_layers = 7
    num_heads = 8
    feedforward_dim = 3200
    dropout = 0.1
    
    # CVAE settings
    latent_dim = 32
    beta = 10  # weight for KL divergence loss
    
    # Training settings
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-5
    weight_decay = 1e-4
    clip_grad_norm = 1.0
    
    # Scheduler settings
    use_scheduler = True
    scheduler_step_size = 30
    scheduler_gamma = 0.1
    
    # Evaluation settings
    eval_frequency = 5  # evaluate every N epochs
    num_eval_episodes = 25
    
    # Logging and checkpoints
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
    save_frequency = 10  # save model every N epochs
    
    # Data augmentation
    use_data_augmentation = True
    random_crop = True
    random_flip = True
    color_jitter = True
    
    # Temporal ensembling
    use_temporal_ensemble = True
    ensemble_weight = 0.5  # weight for exponential moving average

config = Config()