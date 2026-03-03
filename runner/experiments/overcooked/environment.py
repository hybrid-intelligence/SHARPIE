import numpy as np
from gymnasium.envs.registration import register
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked as OriginalOvercooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


# -------------------#---
# -- Light wrapper --#---
# -------------------#---


class WrappedOvercooked(OriginalOvercooked):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, *args, render_mode=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_mode = "rgb_array"

    @staticmethod
    def json_obs(obs):
        return tuple(o.astype(np.float32).tolist() for o in obs["both_agent_obs"])

    def reset(self, seed: 'int | None' = None, options: 'dict[str, Any] | None' = None):
        # Just to follow gym API
        obs = super().reset()
        return self.json_obs(obs), obs

    def step(self, actions: dict):  # Convert OvercookedState to dictionary for serialization
        obs, reward, done, info = super().step(actions.values())
        obs["overcooked_state"] = obs["overcooked_state"].to_dict()
        return self.json_obs(obs), reward, done, False, info

    def render(self):  # Convert BGR -> RGB
        return super().render()[:, :, ::-1]


# -------------------#---

register(
    id="Overcooked-v1",
    entry_point="environment:FixedOvercooked",
)

mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
environment = WrappedOvercooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
