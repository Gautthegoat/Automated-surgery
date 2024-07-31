import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import time
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from models.act_model import ACTModel, loss_function
from utils.config import Config as config
from data.dataset import SurgicalRobotDataset

def train():
    # Set device
    device = torch.device(config.device)

    # Initialize dataset and dataloader
    train_dataset = SurgicalRobotDataset(config.data_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    val_dataset = SurgicalRobotDataset(config.data_dir, split='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Initialize model
    model = ACTModel().to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    # Initialize TensorBoard writer
    log_dir = os.path.join(config.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            image, current_joints, future_joints, true_actions = [b.to(device) for b in batch]

            optimizer.zero_grad()

            pred_actions, mu, logvar = model(image, current_joints, future_joints, training=True)
            loss = loss_function(pred_actions, true_actions, mu, logvar)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)  # Important with transformers apparently
            optimizer.step()

            train_loss += loss.item()

            # Log training loss
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1

        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s")  

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image, current_joints, future_joints, true_actions = [b.to(device) for b in batch]
                pred_actions, mu, logvar = model(image, current_joints, future_joints, training=False)
                loss = loss_function(pred_actions, true_actions, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Log epoch results
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config.checkpoint_dir}/best_model.pth")

        # Save checkpoint
        if (epoch + 1) % config.save_frequency == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f"{config.checkpoint_dir}/checkpoint_epoch{epoch+1}.pth")

        scheduler.step()

    writer.close()

if __name__ == "__main__":
    train()