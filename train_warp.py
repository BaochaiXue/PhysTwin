from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import os
import pickle
import json


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame

    if "cloth" in case_name or "package" in case_name:
        yaml_path = "configs/cloth.yaml"
    else:
        yaml_path = "configs/real.yaml"

    # First-order stage loads YAML and then the optimal parameters produced by
    # the zero-order CMA-ES step.
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    cfg.load_first_order_params(yaml_path, optimal_path)

    print(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"experiments/{case_name}"

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
    )
    trainer.train()
