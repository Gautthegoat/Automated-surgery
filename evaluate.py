import torch
from utils.config import Config
from models.act_model import ACTModel
from data.dataset import SurgicalRobotDataset
from utils.transform import Transform
import numpy as np

def evaluate(checkpoint, device, number_of_tests=10):
    """
    Evaluate the model, extracting the average error of each joint.
    """
    # Upload the right environment
    checkpoint = torch.load(checkpoint)
    config = Config()
    config = config.from_dict(checkpoint['config'])
    print(config.stats)
    exit()

    # Initialize dataset and dataloader
    test_dataset = SurgicalRobotDataset(config, Transform, start_crop=210, end_crop=900)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Initialize model
    model = ACTModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    predictions = []
    gt = []

    with torch.no_grad():
        for idx, (frame, logs) in enumerate(test_loader):
            frame, logs = frame.to(device), logs.to(device)
            current_actions = logs[:, 0, :].unsqueeze(1)
            future_actions = logs[:, 1:, :]
            pred_actions, mu, logvar = model(frame, current_actions, future_actions, training=True)
            predictions.append(pred_actions.cpu().numpy())
            gt.append(future_actions.cpu().numpy())
            
            print(f"Step {idx+1}/{len(test_loader)}", end='\r')
            if idx == number_of_tests - 1:
                break

    for i in range(len(predictions[0][0][7])): 
        print(f"Predicted value for joint {i} : {predictions[0][0][7][i]}, GT value: {gt[0][0][7][i]}")

if __name__ == "__main__":
    avg_errors = evaluate("Archive/checkpoints/ACT_b32_meanstd/best_model.pth", "cuda:1", number_of_tests=1)