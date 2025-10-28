

from backend.model_serving import ModelServer

def test_dummy():
    ms = ModelServer('model.pth')
    # cannot run without model + image; this is a placeholder smoke test
    assert ms is not None
