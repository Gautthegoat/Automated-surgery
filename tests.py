import torch
from models import ACTModel
from utils import Config as config
import time

def test_act_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = ACTModel().to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params/1e6:.2f}M")

    batch_size = 1
    image = torch.randn(batch_size, 3, int(1080/3), int(1920/3)).to(device)
    current_joints = torch.randn(batch_size, config.num_joints).to(device)
    future_joints = torch.randn(batch_size, config.chunk_size, config.num_joints).to(device)

    with torch.no_grad():
        action_sequence, mu, logvar = model(image, current_joints, future_joints, training=True)

    expected_action_shape = (batch_size, config.chunk_size, config.action_dim)
    expected_latent_shape = (batch_size, config.latent_dim)

    print(f"")
    print(f"{':]' if action_sequence.shape == expected_action_shape else 'X'} Training action_sequence shape: {'Passed' if action_sequence.shape == expected_action_shape else 'Failed'}. Expected {expected_action_shape}, got: {tuple(action_sequence.shape)}")
    print(f"{':]' if mu.shape == expected_latent_shape else 'X'} Training mu shape: {'Passed' if mu.shape == expected_latent_shape else 'Failed'}. Expected {expected_latent_shape}, got: {tuple(mu.shape)}")
    print(f"{':]' if logvar.shape == expected_latent_shape else 'X'} Training logvar shape: {'Passed' if logvar.shape == expected_latent_shape else 'Failed'}. Expected {expected_latent_shape}, got: {tuple(logvar.shape)}")
    print(f"")

    start_time = time.time()
    with torch.no_grad():
        action_sequence, mu, logvar = model(image, current_joints, training=False)
    print(f"Time taken for inference mode: {time.time() - start_time:.4f} seconds")

    print(f"{':]' if action_sequence.shape == expected_action_shape else 'X'} Inference action_sequence shape: {'Passed' if action_sequence.shape == expected_action_shape else 'Failed'}. Expected {expected_action_shape}, got: {tuple(action_sequence.shape)}")
    print(f"{':]' if mu is None else 'X'} Inference mu is None: {'Passed' if mu is None else 'Failed'}. Expected None, got: {type(mu)}")
    print(f"{':]' if logvar is None else 'X'} Inference logvar is None: {'Passed' if logvar is None else 'Failed'}. Expected None, got: {type(logvar)}")

    print(f"")
    print("All tests completed.")

if __name__ == "__main__":
    test_act_model()