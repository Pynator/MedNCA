import torch

from model.model import CAModel


def convert():
    pytorch_model = CAModel(n_channels=16, hidden_channels=128, fire_rate=0.5, device="cpu")
    pytorch_model.load_state_dict(torch.load("runs/Jun03_09-52-28_x250/model"))
    pytorch_model.eval()
    dummy_input = torch.zeros(size=(1, 16, 28, 28))
    torch.onnx.export(
        pytorch_model, dummy_input, "runs/Jun03_09-52-28_x250/onnx_model", verbose=True
    )
