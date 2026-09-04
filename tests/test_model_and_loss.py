import torch

from method.loss import MultiTaskAsymmetricLoss
from method.model import AMT


def test_amt_output_shape_and_clamp():
    model = AMT(sw_dim=116, skip_dim=9)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(3, 116), torch.zeros(3, 9))
    assert output.shape == (3, 4)
    assert torch.all(output >= -6.5)
    assert torch.all(output <= 4.0)


def test_amt_rejects_wrong_feature_shape():
    model = AMT(sw_dim=116, skip_dim=9)
    try:
        model(torch.zeros(2, 115), torch.zeros(2, 9))
    except ValueError as exc:
        assert "x_sw" in str(exc)
    else:
        raise AssertionError("wrong x_sw width should fail")


def test_asymmetric_loss_penalizes_active_underprediction():
    criterion = MultiTaskAsymmetricLoss((5.0, 50.0, 50.0, 10.0))
    target = torch.tensor([[-4.0, -4.0, -4.0, -4.0]])
    under = torch.tensor([[-5.0, -5.0, -5.0, -5.0]])
    over = torch.tensor([[-3.0, -3.0, -3.0, -3.0]])
    assert criterion(under, target) > criterion(over, target)
