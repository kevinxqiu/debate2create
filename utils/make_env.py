# utils/make_env.py
from pathlib import Path
from envs.ant import Ant
from envs.half_cheetah import Halfcheetah
from envs.hopper import Hopper
from envs.swimmer import Swimmer
from envs.walker2d import Walker2d

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def make_env(cfg, reward_fn=None, custom_xml_path=None):
    try:
        env_name = cfg["env"]["env_name"] if isinstance(cfg, dict) else cfg.env.env_name
    except (KeyError, AttributeError) as e:
        raise ValueError(f"Could not find env_name in config: {e}")

    # Use custom XML path if provided, otherwise use config
    if custom_xml_path is not None:
        xml_path = Path(custom_xml_path).expanduser()
    else:
        try:
            xml_cfg = cfg["env"]["xml_path"] if isinstance(cfg, dict) else cfg.env.xml_path
            xml_path = Path(xml_cfg).expanduser()
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Could not find xml_path in config: {e}")

    if not xml_path.is_absolute():
        xml_path = (PROJECT_ROOT / xml_path).resolve()
    else:
        xml_path = xml_path.resolve()

    if not xml_path.exists():
        raise FileNotFoundError(f"Could not find xml at: {xml_path}")

    # Create environment based on env_name
    env_name_lower = env_name.lower()
    if env_name_lower == "ant":
        return Ant(xml_path=str(xml_path), reward_fn=reward_fn)
    elif env_name_lower == "half_cheetah":
        return Halfcheetah(xml_path=str(xml_path), reward_fn=reward_fn)
    elif env_name_lower == "hopper":
        return Hopper(xml_path=str(xml_path), reward_fn=reward_fn)
    elif env_name_lower == "swimmer":
        return Swimmer(xml_path=str(xml_path), reward_fn=reward_fn)
    elif env_name_lower == "walker2d":
        return Walker2d(xml_path=str(xml_path), reward_fn=reward_fn)
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
