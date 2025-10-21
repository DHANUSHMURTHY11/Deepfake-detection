import random

def deepfake_detector(image_path):
    """
    Mock deepfake detection function.
    Returns:
        - result: 'fake' or 'real'
        - confidence: confidence percentage
    """
    confidence = random.uniform(85, 99)  # Random confidence score
    result = random.choice(['real', 'fake'])  # Random result

    return result, round(confidence, 2)
