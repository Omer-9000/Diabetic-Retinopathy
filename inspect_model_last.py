import torch

try:
    checkpoint = torch.load(r'c:\Users\AK\Documents\dr\dr_finetuned.pth', map_location='cpu')
    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
    
    layer_names = list(state_dict.keys())
    print("\nLast 5 layers:")
    for key in layer_names[-5:]:
        print(f"{key}: {state_dict[key].shape}")

except Exception as e:
    print(f"Error: {e}")
