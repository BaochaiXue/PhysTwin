# PhysTwin Configuration Parameters

This document describes the parameters used in PhysTwin's YAML configuration files (`configs/cloth.yaml` and `configs/real.yaml`).
Each parameter's default value comes from `qqtt/utils/config.py`. References to the
code and paper are provided where relevant.

## Simulation Settings

| Parameter | Default | Type | Recommended Range | Description & Impact |
|-----------|---------|------|-------------------|---------------------|
| `data_type` | `"real"` | str | `"real"` or `"synthetic"` | Determines which dataset loader is used in `trainer_warp.py` and affects all subsequent processing. |
| `FPS` | `30` | int | `>=1` | Frames per second. Together with `dt`, it defines the number of substeps per frame. |
| `dt` | `5e-5` | float | small positive | Simulation time step. Appears in the explicit Euler update equation in the paper. |
| `num_substeps` | `round(1.0 / FPS / dt)` | int | depends on `dt` and `FPS` | Number of simulation steps per frame. Used throughout the simulator when iterating over substeps. |
| `dashpot_damping` | `100` | float | `0–200` | Dashpot damping coefficient $γ$ in Eq. (6) of the paper. Controls energy dissipation. |
| `drag_damping` | `3` | float | `0–20` | Drag damping $δ$ in the state update equation. Higher values slow velocities. |
| `base_lr` | `1e-3` | float | `1e-4–1e-2` | Learning rate for the Adam optimizer in `trainer_warp.py`. |
| `iterations` | `250` | int | `>0` | Number of training iterations. Determines how many epochs are run. |
| `vis_interval` | `10` | int | `>=1` | Interval for saving visualization videos during training. |
| `init_spring_Y` | `3000` | float | `spring_Y_min–spring_Y_max` | Initial spring stiffness (Young's modulus) passed to the simulator. |

## Collision Parameters

| Parameter | Default | Type | Recommended Range | Description & Impact |
|-----------|---------|------|-------------------|---------------------|
| `collide_elas` | `0.5` | float | `0–1` | Restitution for ground collisions. Affects velocity after impact in `integrate_ground_collision`. |
| `collide_fric` | `0.3` | float | `0–2` | Friction coefficient for ground collisions. |
| `collide_object_elas` | `0.7` | float | `0–1` | Restitution when objects collide with each other. |
| `collide_object_fric` | `0.3` | float | `0–2` | Friction for object–object collisions. |
| `collision_dist` | `0.06` | float | `0.01–0.05` | Threshold distance to detect potential collisions. |
| `collision_learn` | `True` | bool | n/a | If true, collision parameters are optimized during training. |
| `self_collision` | `False` | bool | n/a | When enabled, each vertex is considered for collision against all others, increasing computation. |

## Topology & Control

| Parameter | Default | Type | Recommended Range | Description & Impact |
|-----------|---------|------|-------------------|---------------------|
| `object_radius` | `0.02` | float | `0.01–0.05` | Neighborhood radius used to connect springs between object points during initialization. |
| `object_max_neighbours` | `30` | int | `10–50` | Maximum neighbors connected per point. |
| `controller_radius` | `0.04` | float | `0.01–0.08` | Radius to connect control points to nearby object points. |
| `controller_max_neighbours` | `50` | int | `10–80` | Maximum number of connections per control point. |
| `spring_Y_min` | `0` | float | min stiffness | Lower bound when clamping spring stiffness. |
| `spring_Y_max` | `1e5` | float | max stiffness | Upper bound when clamping spring stiffness. |

## Visualization & Camera

| Parameter | Default | Type | Recommended Range | Description & Impact |
|-----------|---------|------|-------------------|---------------------|
| `reverse_z` | `True` | bool | n/a | Reverses the direction of gravity; useful when coordinate frames differ. |
| `vp_front` | `[1,0,-2]` | list(float) | user defined | Front vector for the default Open3D viewpoint. |
| `vp_up` | `[0,0,-1]` | list(float) | user defined | Up vector for the default viewpoint. |
| `vp_zoom` | `1` | float | `>0` | Zoom factor for visualization. |

## Loss Weights

| Parameter | Default | Type | Description & Impact |
|-----------|---------|------|---------------------|
| `chamfer_weight` | `1.0` | float | Weight for the chamfer distance in `calculate_loss`. |
| `track_weight` | `1.0` | float | Weight for tracking loss in `calculate_loss`. |

These parameters form the core configuration for PhysTwin.
They correspond to the physical and optimization variables discussed in the paper, particularly the hierarchical optimization strategy that refines spring stiffness and collision parameters.
