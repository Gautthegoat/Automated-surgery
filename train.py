import torch
from torch.optim import AdamW
import time
import os
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from models.act_model import ACTModel, loss_function
from utils.config import Config
from utils.transform import Transform
from data.dataset import SurgicalRobotDataset

def train():
    # Set device
    config = Config()
    device = torch.device(config.device)

    # Initialize dataset and dataloader
    train_dataset = SurgicalRobotDataset(config, Transform, start_crop=210, end_crop=900)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # Initialize model
    model = ACTModel(config).to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Initialize TensorBoard writer
    log_dir = os.path.join(config.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Training loop
    best_train_loss = float('inf')
    global_step = 0
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for idx, (frame, logs) in enumerate(train_loader):
            frame, logs = frame.to(device), logs.to(device)
            current_actions = logs[:, 0, :].unsqueeze(1) # (batch_size, 1, num_joints)
            future_actions = logs[:, 1:, :] # (batch_size, chunk_size, num_joints)

            pred_actions, mu, logvar = model(frame, current_actions, future_actions, training=True)
            loss = loss_function(pred_actions, future_actions, mu, logvar, config)

            optimizer.zero_grad()
            loss.backward()
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)  # Could be important with transformers apparently
            optimizer.step()

            train_loss += loss.item()

            global_step += 1
            print(f"Epoch {epoch+1}/{config.num_epochs}: Step {idx+1}/{len(train_loader)}, loss: {loss.item()}", end='\r')

        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {train_loss:.4f}, Time: {epoch_time/60:.2f}min, lr: {optimizer.param_groups[0]['lr']}")  

        # Log epoch results
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)

        # Save best model
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save({
                'config': config.to_dict(),
                'epoch': epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            },  f"{config.checkpoint_dir}/best_model.pth")

        # Save checkpoint
        if (epoch + 1) % config.save_frequency == 0:
            torch.save({
                'config': config.to_dict(),
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, f"{config.checkpoint_dir}/epoch_{epoch+1}.pth")

    writer.close()

if __name__ == "__main__":
    train()