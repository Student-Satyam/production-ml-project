from src.model import predict_price


def test_prediction():
    prediction = predict_price(2200)

    assert prediction is not None


def test_prediction_is_positive():
    prediction = predict_price(2200)

    assert prediction > 0