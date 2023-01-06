import pytest
from table15.utils import pipeline_utils as plutils


@pytest.fixture
def test_configs():
    yaml_path = "/Users/ag46548/dev/github/KaleRP/table15/tests/configs/test_configs.yaml"
    configs = plutils.yaml_parser(yaml_path)
    
    assert "CONFIGS" in configs
    assert "HYPERPARAMS" in configs
    assert "FEATURES" in configs
    assert "MODELS" in configs
    
    return configs
    
def test_yaml_parser(test_configs):
    assert test_configs["CONFIGS"]["POLICY"] == "test"

def test_get_from_configs(test_configs):
    
    assert plutils.get_from_configs(test_configs, "POLICY", param_type="CONFIGS") == "test"
    assert plutils.get_from_configs(test_configs, "policy", param_type="CONFIGS") == "test"
    assert plutils.get_from_configs(test_configs, "foo", param_type="CONFIGS") == None  
    
