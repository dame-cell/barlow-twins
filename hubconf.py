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
    model = _resnet50(pretrained=False, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/barlowtwins/ljng/checkpoint.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model
