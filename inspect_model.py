import torch

try:
    checkpoint = torch.load(r'c:\Users\AK\Documents\dr\dr_finetuned.pth', map_location='cpu')
    if isinstance(checkpoint, dict):
        print("Keys in checkpoint:", checkpoint.keys())
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint

    print("\nFirst 10 layers in state dict:")
    for i, key in enumerate(state_dict.keys()):
        if i >= 10: break
        print(f"{key}: {state_dict[key].shape}")

    # Try to infer architecture
    if any('resnet' in key.lower() for key in state_dict.keys()):
        print("\nLikely architecture: ResNet")
    elif any('features.0.weight' in key for key in state_dict.keys()):
        print("\nLikely architecture: Sequential/Mobilenet/VGG style")
    
except Exception as e:
    print(f"Error loading model: {e}")
