import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


class Transform:
    def transform_frame(height, width, frame):
        """
        Resize, normalize and transform an array to tensor.
        """
        transform = A.Compose([
            A.Resize(height=height, width=width),  # Resize to standard ImageNet size
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
                max_pixel_value=255.0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        transformed = transform(image=frame)

        return transformed['image']  # This is now a torch.Tensor of shape (3, 224, 224)

    def transform_log(log):
        """
        Transform a log array to a tensor.
        (nb_logs,) -> (1, nb_logs)
        """
        log_tensor = torch.from_numpy(log).unsqueeze(0).float()

        return log_tensor

    # Example usage:
    if __name__ == "__main__":
        # Example frame (replace with your actual frame)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Example log (replace with your actual log)
        log = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
        transformed_frame = transform_frame(frame)
        transformed_log = transform_log(log)
        
        print("Transformed frame shape:", transformed_frame.shape)
        print("Transformed frame dtype:", transformed_frame.dtype)
        print("Transformed log shape:", transformed_log.shape)
        print("Transformed log dtype:", transformed_log.dtype)