import torch
import numpy as np
from tqdm import tqdm
from models.act_model import loss_function

def evaluate(model, test_loader, device, temporal_agg=True, num_queries=100):
    model.eval()
    total_loss = 0.0
    all_pred_actions = []
    all_true_actions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            image, current_joints, future_joints, true_actions = [b.to(device) for b in batch]
            
            if temporal_agg:
                # Initialize tensor to store all predicted actions
                all_time_actions = torch.zeros(true_actions.shape[0], true_actions.shape[1] + num_queries, true_actions.shape[2]).to(device)
                
                for t in range(true_actions.shape[1]):
                    # Predict actions for the next num_queries steps
                    pred_actions, _, _ = model(image, current_joints[:, t:t+1], future_joints[:, t:t+num_queries], training=False)
                    
                    # Store predicted actions
                    all_time_actions[:, t:t+num_queries] += pred_actions

                    # Apply exponential weighting
                    k = 0.01
                    exp_weights = torch.exp(-k * torch.arange(num_queries)).to(device)
                    exp_weights = exp_weights / exp_weights.sum()
                    
                    # Compute weighted average of predictions
                    actions_for_curr_step = all_time_actions[:, t:t+num_queries]
                    weighted_actions = (actions_for_curr_step * exp_weights[None, :, None]).sum(dim=1, keepdim=True)
                    
                    # Update prediction for current timestep
                    all_time_actions[:, t:t+1] = weighted_actions

                pred_actions = all_time_actions[:, :true_actions.shape[1]]
            else:
                pred_actions, _, _ = model(image, current_joints, future_joints, training=False)

            loss = loss_function(pred_actions, true_actions, torch.zeros_like(pred_actions), torch.zeros_like(pred_actions))
            total_loss += loss.item()

            all_pred_actions.append(pred_actions.cpu().numpy())
            all_true_actions.append(true_actions.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    all_pred_actions = np.concatenate(all_pred_actions, axis=0)
    all_true_actions = np.concatenate(all_true_actions, axis=0)

    # Compute MSE loss
    mse_loss = np.mean((all_pred_actions - all_true_actions)**2)
    
    # Compute average error for each joint
    joint_errors = {}
    for i in range(all_pred_actions.shape[-1]):
        joint_errors[f'joint_{i}'] = np.mean(np.abs(all_pred_actions[..., i] - all_true_actions[..., i]))
    
    return avg_loss, mse_loss, joint_errors, all_pred_actions, all_true_actions