from .misc import singleton
import logging
import yaml
import pickle


@singleton
class Config:
    def __init__(self):
        self.data_type = "real"
        self.FPS = 30
        self.dt = 5e-5
        self.num_substeps = round(1.0 / self.FPS / self.dt)

        self.dashpot_damping = 100
        self.drag_damping = 3
        self.base_lr = 1e-3
        self.iterations = 250
        self.vis_interval = 10
        self.init_spring_Y = 3e3
        self.collide_elas = 0.5
        self.collide_fric = 0.3
        self.collide_object_elas = 0.7
        self.collide_object_fric = 0.3

        self.object_radius = 0.02
        self.object_max_neighbours = 30
        self.controller_radius = 0.04
        self.controller_max_neighbours = 50

        self.spring_Y_min = 0
        self.spring_Y_max = 1e5

        self.reverse_z = True
        self.vp_front = [1, 0, -2]
        self.vp_up = [0, 0, -1]
        self.vp_zoom = 1

        self.collision_dist = 0.06
        # Parameters on whether update the collision parameters
        self.collision_learn = True
        self.self_collision = False

        # DEBUG mode: set use_graph to False
        self.use_graph = True

        # Attribute for the real
        self.chamfer_weight = 1.0
        self.track_weight = 1.0
        self.acc_weight = 0.01

        # Other parameters for visualization
        self.overlay_path = None

        # track the last loaded pickle to avoid re-reading the file if the same
        # parameters are requested repeatedly.  This helps when zero-order and
        # first-order stages run in the same Python process.
        self._loaded_optimal_path: str | None = None

    def to_dict(self):
        # Convert the class to dictionary
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                setattr(self, key, value)

    def load_from_yaml(self, file_path):
        with open(file_path) as file:
            config_dict = yaml.safe_load(file)
        self.update_from_dict(config_dict)

    def load_from_yaml_with_optimal(
        self, yaml_path: str, optimal_path: str | None = None, use_global_spring_Y: bool = True
    ) -> None:
        """Load base parameters from YAML and optionally override with optimal pickle.

        Previously the calling scripts loaded the YAML and then loaded the
        ``optimal_params.pkl`` separately. When both zero-order and first-order
        stages were executed in sequence, ``optimal_params.pkl`` ended up being
        loaded twice.  This helper ensures that the file is loaded exactly once
        and clearly separates the two stages: pass ``optimal_path`` only for the
        first-order optimization step.
        """

        if optimal_path is None:
            # Zero-order stage
            self.load_zero_order_params(yaml_path)
            return

        # First-order stage
        self.load_first_order_params(
            yaml_path,
            optimal_path,
            use_global_spring_Y=use_global_spring_Y,
        )

    # ------------------------------------------------------------------
    # New helpers to emphasise the two-stage loading process
    # ------------------------------------------------------------------
    def load_zero_order_params(self, yaml_path: str) -> None:
        """Load parameters solely from ``yaml_path`` for zero-order CMA-ES.

        The method also resets ``_loaded_optimal_path`` so that the subsequent
        first-order optimization can safely call :func:`load_first_order_params`
        without thinking that the pickle has already been processed.
        """

        self.load_from_yaml(yaml_path)
        self._loaded_optimal_path = None

    def load_first_order_params(
        self,
        yaml_path: str,
        optimal_path: str,
        *,
        use_global_spring_Y: bool = True,
    ) -> None:
        """Load parameters for gradient-based refinement.

        ``optimal_path`` is cached to avoid redundant loading.  Previously the
        caller would load the YAML file and then re-apply ``optimal_params.pkl``
        each time, which reset ``init_spring_Y`` back to the YAML value before
        reading the pickle again.  By checking ``_loaded_optimal_path`` we ensure
        that repeated invocations are idempotent within a single process.
        """

        if optimal_path == self._loaded_optimal_path:
            # Avoid re-reading both YAML and pickle if nothing changed
            return

        self.load_from_yaml(yaml_path)

        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)

        self.set_optimal_params(optimal_params, use_global_spring_Y=use_global_spring_Y)
        self._loaded_optimal_path = optimal_path

        # Log the resulting spring stiffness for verification
        logging.info(f"Config init_spring_Y loaded: {self.init_spring_Y}")

    def set_optimal_params(self, optimal_params, use_global_spring_Y=True):
        if use_global_spring_Y:
            optimal_params["init_spring_Y"] = optimal_params.pop("global_spring_Y")
        else:
            optimal_params["init_spring_Y"] = self.init_spring_Y
        self.update_from_dict(optimal_params)


cfg = Config()
