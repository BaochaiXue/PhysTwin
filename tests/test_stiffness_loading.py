import pickle
import importlib.util
import types
import sys

torch_stub = types.ModuleType("torch")
torch_stub.distributed = types.ModuleType("distributed")
sys.modules.setdefault("torch", torch_stub)

pkg = types.ModuleType("qqtt")
pkg.__path__ = ["qqtt"]
sys.modules["qqtt"] = pkg
utils_pkg = types.ModuleType("qqtt.utils")
utils_pkg.__path__ = ["qqtt/utils"]
sys.modules["qqtt.utils"] = utils_pkg

spec_misc = importlib.util.spec_from_file_location(
    "qqtt.utils.misc", "qqtt/utils/misc.py"
)
misc_mod = importlib.util.module_from_spec(spec_misc)
sys.modules["qqtt.utils.misc"] = misc_mod
spec_misc.loader.exec_module(misc_mod)

spec = importlib.util.spec_from_file_location(
    "qqtt.utils.config", "qqtt/utils/config.py"
)
config = importlib.util.module_from_spec(spec)
sys.modules["qqtt.utils.config"] = config
config.__package__ = "qqtt.utils"
spec.loader.exec_module(config)
cfg = config.cfg

def test_stiffness_loading(tmp_path):
    # zero-order stage: only load YAML
    cfg.load_zero_order_params('configs/real.yaml')
    yaml_stiffness = cfg.init_spring_Y
    assert isinstance(yaml_stiffness, float)

    # create a pickle with different stiffness
    new_val = yaml_stiffness + 10.0
    pkl_path = tmp_path / 'opt.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({'global_spring_Y': new_val}, f)

    # first-order stage: load YAML and override with pickle
    cfg.load_first_order_params('configs/real.yaml', str(pkl_path))
    assert cfg.init_spring_Y == new_val

    # loading again should not change the value
    cfg.load_first_order_params('configs/real.yaml', str(pkl_path))
    assert cfg.init_spring_Y == new_val
