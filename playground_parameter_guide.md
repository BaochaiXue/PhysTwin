# Interactive Playground Parameter Guide

This guide describes how to use `interactive_playground.py` to explore a pre-built PhysTwin.
It lists all command-line options, explains their effects, and provides recommended settings
for common scenarios.

## Parameter Overview

| Option | Default | Description |
|-------|---------|-------------|
| `--base_path` | `./data/different_types` | Directory containing processed data for each case. Each case must reside in its own subfolder. |
| `--gaussian_path` | `./gaussian_output` | Location of the static Gaussian Splatting results. The loader expects `<case_name>/<exp_name>/point_cloud/iteration_10000/point_cloud.ply` under this directory. |
| `--bg_img_path` | `./data/bg.png` | Background image blended with the rendered object. |
| `--case_name` | `double_lift_cloth_3` | Name of the case to load. Available cases are listed in `data_config.csv`. |
| `--n_ctrl_parts` | `2` | Number of controller sets. Use `1` for single-hand control or `2` for dual-hand. |
| `--inv_ctrl` | disabled | If set, horizontal motion keys are inverted. |

## Detailed Explanations

### `--base_path`
The playground loads calibration, metadata and preprocessed simulation data from
`<base_path>/<case_name>`. Set this path to wherever you stored the downloaded
or processed datasets.

### `--gaussian_path`
Path to the Gaussian Splatting model that provides realistic appearance. A
subdirectory for each case should match the structure used during training. If
you trained the Gaussian model elsewhere, point `--gaussian_path` to that
location.

### `--bg_img_path`
Background image displayed behind the rendered object. Replace this with a plain
color image or any other scene for different visual styles.

### `--case_name`
Identifier of the PhysTwin you want to interact with. Ensure the same case name
exists under both `--base_path` and `--gaussian_path`. Check `data_config.csv`
for valid names such as `double_stretch_sloth` or `single_push_rope_1`.

### `--n_ctrl_parts`
Specifies how many control panels are active. With `1`, only the `WASDQE` keys
control the object. With `2`, a second set (`IJKLUO`) becomes available.
Control points are automatically clustered between the left and right hand when
using two parts.

### `--inv_ctrl`
Flips the left/right direction of all horizontal movements. Enable this flag if
the camera view is mirrored or you prefer reversed controls.

## Usage Examples

Run the sloth example with two hands:
```bash
python interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_sloth
```

Use a custom data location and invert controls:
```bash
python interactive_playground.py \
    --base_path /my/data/different_types \
    --gaussian_path /my/gaussian_output \
    --inv_ctrl --case_name double_lift_cloth_3
```

Run inside the provided Docker container:
```bash
./docker_scripts/run.sh /path/to/data /path/to/experiments \
                        /path/to/experiments_optimization /path/to/gaussian_output
# inside the container
conda activate phystwin_env
python interactive_playground.py --case_name double_stretch_sloth
```

## Best Practices and Recommendations

- Verify that your chosen `case_name` exists under both data and Gaussian paths.
  Consult `data_config.csv` for the list of supported cases.
- For most cloth or stuffed-animal scenes, two hands (`--n_ctrl_parts 2`) provide
  a natural interaction experience.
- Use `--inv_ctrl` if the scene appears mirrored or left/right feels reversed.
- Keep all required folders (`data`, `experiments`, `experiments_optimization`,
  `gaussian_output`) in consistent locations so paths remain predictable.

With these options configured, `interactive_playground.py` lets you manipulate
a PhysTwin model in real time using simple keyboard controls.
