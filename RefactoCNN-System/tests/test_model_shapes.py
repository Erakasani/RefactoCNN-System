import torch
from src.refactocnn.models.refactocnn import RefactoCNN

def test_forward_shape():
    model = RefactoCNN(input_dim=128)
    x = torch.randn(4, 128)
    y = model(x)
    assert y.shape == (4, 2)
