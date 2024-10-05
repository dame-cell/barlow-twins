import torch
from torchvision.models.resnet import resnet50 as _resnet50

dependencies = ['torch', 'torchvision']

def resnet50(pretrained=True, **kwargs):
    """
    Load ResNet-50 model with Barlow Twins pretrained weights.

    Args:
        pretrained (bool): If True, loads the pretrained weights.
        **kwargs: Additional arguments passed to the model.

    Returns:
        model: The ResNet-50 model with loaded weights.
    """
    model = _resnet50(weights=None, **kwargs)  # Use weights=None to prevent any default loading
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/barlowtwins/ljng/checkpoint.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')

        # Filter the state_dict to only include model weights
        model_state_dict = {}
        for k, v in state_dict['model'].items():
            if k.startswith('module.') or k.startswith('backbone.'):
                model_state_dict[k.replace('module.', '').replace('backbone.', '')] = v
            elif 'optimizer' not in k:  # Exclude optimizer states
                model_state_dict[k] = v  # Keep other keys that might be relevant

        # Load the filtered weights into the model
        model.load_state_dict(model_state_dict, strict=False)
    return model
