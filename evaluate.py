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
    
    print(f"{'Joint':^5} {'Ground Truth t0':^14} {'Ground Truth t1':^14} {'Prediction t1':^15} {'Ground Truth t10':^14} {'Prediction t10':^15} {'Ground Truth t50':^14} {'Prediction t50':^15}") 
    print(f"{'-'*5} {'-'*15} {'-'*15} {'-'*16} {'-'*15} {'-'*16} {'-'*15} {'-'*16}")

    for i in range(len(predictions[0][0][0])):
        gt_0 = gt[0][0][0][i]
        gt_1 = gt[0][0][0][i]
        pred_t1 = predictions[0][0][0][i]
        gt_10 = gt[0][0][9][i]
        pred_t10 = predictions[0][0][9][i]
        gt_50 = gt[0][0][49][i]
        pred_t50 = predictions[0][0][49][i]
        
        print(f"{i:^5} {gt_0:^15.2f} {gt_1:^15.2f} {pred_t1:^16.2f} {gt_10:^15.2f} {pred_t10:^16.2f} {gt_50:^15.2f} {pred_t50:^16.2f}")
if __name__ == "__main__":
    avg_errors = evaluate("Archive/checkpoints/ACT_b32_meanstd/best_model.pth", "cuda:1", number_of_tests=1)