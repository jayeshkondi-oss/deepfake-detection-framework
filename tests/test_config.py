from deepfake.utils.config import load_config


def test_config_loading():
    config = load_config()

    assert config.project["name"] == "Multi-Modal Deepfake Detection Framework"
    assert config.project["version"] == "1.0"


def test_data_config():
    config = load_config()

    assert config.data["image_size"] == 224
    assert config.data["sample_frames"] == 16
    assert config.data["audio_sample_rate"] == 16000


def test_model_config():
    config = load_config()

    assert config.model["visual_encoder"] == "vit"
    assert config.model["audio_encoder"] == "wav2vec2"
    assert config.model["fusion"] == "cross_attention"
    assert config.model["temporal_model"] == "transformer"
    assert config.model["num_classes"] == 2


def test_training_config():
    config = load_config()

    assert config.training["batch_size"] == 4
    assert config.training["epochs"] == 30
    assert config.training["learning_rate"] == 0.0001
    assert config.training["mixed_precision"] is True