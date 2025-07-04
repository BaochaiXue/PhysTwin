# Interactive Playground Parameter Guide

This guide explains how to configure `interactive_playground.py` for experimenting with a prebuilt PhysTwin. The playground loads a trained PhysTwin model and the associated Gaussian Splatting representation so you can interactively control the object in real time.

The Python code in this repository follows the standard type hinting syntax introduced in **Python&nbsp;3.10**.  You will therefore see annotations using built‑in generics such as `list[int]` and unions written with the `|` operator.  These hints are purely for readability and static analysis and do not affect the runtime behaviour of the scripts.

## Parameter Overview

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_path` | `./data/different_types` | Root folder containing all processed cases. Each case resides in a subfolder named by its `case_name`. |
| `--gaussian_path` | `./gaussian_output` | Directory with rendered Gaussian Splatting results. The script expects to find the per-case point cloud under `<gaussian_path>/<case_name>/.../point_cloud.ply>`. |
| `--bg_img_path` | `./data/bg.png` | Background image blended with the rendering in the playground window. |
| `--case_name` | `double_lift_cloth_3` | Name of the case to load. All available cases can be found inside `data/different_types` or `data_config.csv`. |
| `--n_ctrl_parts` | `2` | Number of control parts (hands). Use `1` for single‑hand or `2` for double‑hand control. |
| `--inv_ctrl` | `False` | Invert the horizontal movement direction when pressing control keys. Useful when the camera faces the scene from the opposite direction. |
| `--ignore_checkpoint_stiffness` | `False` | Keep `init_spring_Y` from the YAML instead of the trained checkpoint. |

## Detailed Explanations

### `--base_path`
Path to the dataset folder that stores processed data for every case. The playground reads `calibrate.pkl`, `metadata.json` and `final_data.pkl` from `<base_path>/<case_name>`. Change this if your data is kept elsewhere.

### `--gaussian_path`
Location of the static Gaussian Splatting model used for visualization. If you trained or downloaded the Gaussian results in another directory, set this path accordingly.

### `--bg_img_path`
Image file blended as the background during rendering. Replace it with your own image (for example a blank color or another scene) for different visual effects.

### `--case_name`
Chooses which PhysTwin case to load. The repository provides various cases such as `double_stretch_sloth`, `single_push_rope_1`, etc. Ensure the same name exists under both `--base_path` and `--gaussian_path` directories.

### `--n_ctrl_parts`
Controls how many sets of keyboard commands are active. With `1`, only the `WASDQE` keys move a single controller. With `2`, both `WASDQE` (left hand) and `IJKLUO` (right hand) are enabled. The positions of control points are automatically clustered for two‑handed control.

### `--inv_ctrl`
Flips the sign of horizontal motion for all keys. Enable this flag if you want the object to move left when pressing `D`/`L` and right when pressing `A`/`J`, which can be convenient if the camera orientation is mirrored.

## Usage Examples

Interact with the sloth with two hands:
```bash
python interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_sloth
```

Control a cloth with inverted horizontal keys:
```bash
python interactive_playground.py --inv_ctrl --n_ctrl_parts 2 --case_name double_lift_cloth_3
```

Run in a Docker container with custom data locations:
```bash
./docker_scripts/run.sh /path/to/data /path/to/experiments /path/to/experiments_optimization /path/to/gaussian_output
# Inside the container
conda activate phystwin_env
python interactive_playground.py --case_name double_stretch_sloth
```

## Best Practices and Recommendations

- **Check case availability**: The list of usable case names is stored in [`data_config.csv`](./data_config.csv). Make sure the chosen name exists under your `--base_path` and `--gaussian_path`.
- **Use two control parts for dual‑hand setups**: Most cloth or stuffed animal cases rely on two hands; set `--n_ctrl_parts 2` for natural interaction.
- **Invert control when the camera view is flipped**: If left/right movements feel reversed due to your camera configuration, add `--inv_ctrl`.
- **Keep data paths organized**: Place `data`, `experiments_optimization`, `experiments`, and `gaussian_output` in predictable directories and point `--base_path` and `--gaussian_path` to them for hassle‑free runs.

With these parameters tuned to your setup, `interactive_playground.py` allows real‑time exploration of each PhysTwin model using intuitive keyboard controls.

