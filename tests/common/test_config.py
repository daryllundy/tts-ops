import pytest
from pydantic import ValidationError

from common.config import TTSServiceSettings


class TestConfig:

    def test_device_validation_valid(self):
        """
        Property: Configuration validation consistency - Valid inputs
        """
        valid_devices = ["auto", "mps", "cpu", "cuda:0", "cuda:1"]
        for device in valid_devices:
            settings = TTSServiceSettings(device=device)
            assert settings.device == device

    def test_device_validation_invalid(self):
        """
        Property: Configuration validation consistency - Invalid inputs
        """
        invalid_devices = ["invalid", "gpu", "", "cuda:", "cuda:abc", "cuda:-1"]
        for device in invalid_devices:
            with pytest.raises(ValidationError) as excinfo:
                TTSServiceSettings(device=device)
            assert "Device must be" in str(excinfo.value)

    def test_default_auto_selection(self):
        """
        Property: Default configuration auto-selection
        """
        # Ensure default is "auto"
        settings = TTSServiceSettings()
        assert settings.device == "auto"

    def test_env_var_device_selection(self, monkeypatch):
        """
        Property: Environment variable device selection
        """
        monkeypatch.setenv("TTS_DEVICE", "mps")
        settings = TTSServiceSettings()
        assert settings.device == "mps"

        monkeypatch.setenv("TTS_DEVICE", "cpu")
        settings = TTSServiceSettings()
        assert settings.device == "cpu"
