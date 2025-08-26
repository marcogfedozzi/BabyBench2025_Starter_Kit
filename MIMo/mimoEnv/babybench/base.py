""" Base class for all BabyBench environments.
"""
import os
import numpy as np
import mujoco
import pickle
import time

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY
from mimoActuation.actuation import SpringDamperModel, PositionalModel
from mimoActuation.muscle import MuscleModel
import mimoEnv.utils as env_utils
from copy import deepcopy


SCENE_XML = os.path.join(SCENE_DIRECTORY, "babybench.xml")
""" Path to the stand up scene.

:meta hide-value:
"""

VESTIBULAR_PARAMS = {
    "sensors": ["vestibular_acc", "vestibular_gyro"],
}
""" Default vestibular parameters.

:meta hide-value:
"""


PROPRIOCEPTION_PARAMS = {
    "components": ["velocity", "torque", "limits", "actuation"],
    "threshold": .035,
}
""" Default parameters for proprioception. Relative joint positions are always included.

:meta hide-value:
"""

TOUCH_PARAMS = {
    "scales": {
        "left_toes": 0.010,
        "right_toes": 0.010,
        "left_foot": 0.015,
        "right_foot": 0.015,
        "left_lower_leg": 0.038,
        "right_lower_leg": 0.038,
        "left_upper_leg": 0.027,
        "right_upper_leg": 0.027,
        "hip": 0.025,
        "lower_body": 0.025,
        "upper_body": 0.030,
        "head": 0.013,
        "left_eye": 1.0,
        "right_eye": 1.0,
        "left_upper_arm": 0.024,
        "right_upper_arm": 0.024,
        "left_lower_arm": 0.024,
        "right_lower_arm": 0.024,
        "left_hand": 0.007,
        "right_hand": 0.007,
        "left_fingers": 0.002,
        "right_fingers": 0.002,
    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}
""" Default touch parameters.

:meta hide-value:
"""

TOUCH_PARAMS_V2 = {
    "scales": {
        "left_big_toe": 0.010,
        "right_big_toe": 0.010,
        "left_toes": 0.010,
        "right_toes": 0.010,
        "left_foot": 0.015,
        "right_foot": 0.015,
        "left_lower_leg": 0.038,
        "right_lower_leg": 0.038,
        "left_upper_leg": 0.027,
        "right_upper_leg": 0.027,
        "hip": 0.025,
        "lower_body": 0.025,
        "upper_body": 0.030,
        "head": 0.013,
        "left_eye": 1.0,
        "right_eye": 1.0,
        "left_upper_arm": 0.024,
        "right_upper_arm": 0.024,
        "left_lower_arm": 0.024,
        "right_lower_arm": 0.024,
        "left_hand": 0.007,
        "right_hand": 0.007,
        "left_ffdistal": 0.002,
        "left_mfdistal": 0.002,
        "left_rfdistal": 0.002,
        "left_lfdistal": 0.002,
        "left_thdistal": 0.002,
        "left_ffmiddle": 0.004,
        "left_mfmiddle": 0.004,
        "left_rfmiddle": 0.004,
        "left_lfmiddle": 0.004,
        "left_thhub": 0.004,
        "left_ffknuckle": 0.004,
        "left_mfknuckle": 0.004,
        "left_rfknuckle": 0.004,
        "left_lfknuckle": 0.004,
        "left_thbase": 0.004,
        "left_lfmetacarpal": 0.007,
        "right_ffdistal": 0.002,
        "right_mfdistal": 0.002,
        "right_rfdistal": 0.002,
        "right_lfdistal": 0.002,
        "right_thdistal": 0.002,
        "right_ffmiddle": 0.004,
        "right_mfmiddle": 0.004,
        "right_rfmiddle": 0.004,
        "right_lfmiddle": 0.004,
        "right_thhub": 0.004,
        "right_ffknuckle": 0.004,
        "right_mfknuckle": 0.004,
        "right_rfknuckle": 0.004,
        "right_lfknuckle": 0.004,
        "right_thbase": 0.004,
        "right_lfmetacarpal": 0.007,
    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}
""" Default touch parameters for the v2 version of MIMo with five fingers and two toes.

:meta hide-value:
"""

BODY_GROUPS = {
    "left_big_toe": "feet",
    "right_big_toe": "feet",
    "left_toes": "feet",
    "right_toes": "feet",
    "left_foot": "feet",
    "right_foot": "feet",
    "left_lower_leg": "legs",
    "right_lower_leg": "legs",
    "left_upper_leg": "legs",
    "right_upper_leg": "legs",
    "hip": "body",
    "lower_body": "body",
    "upper_body": "body",
    "head": "head",
    "left_eye": "eyes",
    "right_eye": "eyes",
    "left_upper_arm": "arms",
    "right_upper_arm": "arms",
    "left_lower_arm": "arms",
    "right_lower_arm": "arms",
    "left_hand": "hands",
    "right_hand": "hands",
    "left_fingers": "fingers",
    "right_fingers": "fingers",
    "left_ffdistal": "fingers",
    "left_mfdistal": "fingers",
    "left_rfdistal": "fingers",
    "left_lfdistal": "fingers",
    "left_thdistal": "fingers",
    "left_ffmiddle": "fingers",
    "left_mfmiddle": "fingers",
    "left_rfmiddle": "fingers",
    "left_lfmiddle": "fingers",
    "left_thhub": "fingers",
    "left_ffknuckle": "fingers",
    "left_mfknuckle": "fingers",
    "left_rfknuckle": "fingers",
    "left_lfknuckle": "fingers",
    "left_thbase": "fingers",
    "left_lfmetacarpal": "fingers",
    "right_ffdistal": "fingers",
    "right_mfdistal": "fingers",
    "right_rfdistal": "fingers",
    "right_lfdistal": "fingers",
    "right_thdistal": "fingers",
    "right_ffmiddle": "fingers",
    "right_mfmiddle": "fingers",
    "right_rfmiddle": "fingers",
    "right_lfmiddle": "fingers",
    "right_thhub": "fingers",
    "right_ffknuckle": "fingers",
    "right_mfknuckle": "fingers",
    "right_rfknuckle": "fingers",
    "right_lfknuckle": "fingers",
    "right_thbase": "fingers",
    "right_lfmetacarpal": "fingers",
}
""" Grouped body names for MIMo's named body parts.

:meta hide-value:
"""

VISION_PARAMS = {
    "eye_left": {"width": 64, "height": 64},
    "eye_right": {"width": 64, "height": 64},
}
""" Default vision parameters.

:meta hide-value:
"""

ACTUATION_MODELS = {
    'spring_damper': SpringDamperModel,
    'positional': PositionalModel,
    'muscle': MuscleModel,
}

DEFAULT_SIZE = 480
""" Default window size for gym rendering functions.

:meta hide-value:
"""

class BabyBenchEnv(MIMoEnv):
    
    def __init__(self,
                 model_path=SCENE_XML,
                 frame_skip=1,
                 proprio_params=PROPRIOCEPTION_PARAMS,
                 vision_params=VISION_PARAMS,
                 vestibular_params=VESTIBULAR_PARAMS,
                 touch_params=TOUCH_PARAMS,
                 actuation_model=SpringDamperModel,
                 width=DEFAULT_SIZE,
                 height=DEFAULT_SIZE,
                 **kwargs):

        if ('config' in kwargs) and (kwargs['config'] is not None): 
            # Modify default values from config values
            config = kwargs['config']
            if config['vision_active'] is not None:
                if config['vision_active'] is False:
                    vision_params = None
                elif config['vision_resolution'] is not None:
                    vision_params = {
                        "eye_left": {"width": config['vision_resolution'], "height": config['vision_resolution']},
                        "eye_right": {"width": config['vision_resolution'], "height": config['vision_resolution']},
                    }
            if config['vestibular_active'] is not None:
                if config['vestibular_active'] is False:
                    vestibular_params = None
            if config['touch_active'] is not None:
                if (config['touch_active'] is False) or (config['touch_scale'] == 0):
                    touch_params = None
                else:
                    # Build a per-instance touch_params copy and compute scales from the
                    # immutable module-level TOUCH_PARAMS to avoid mutating shared state.
                    base_touch = deepcopy(TOUCH_PARAMS)
                    # Start from a copy of the incoming touch_params if provided, else from base
                    tp = deepcopy(touch_params) if touch_params is not None else deepcopy(base_touch)

                    if config.get('touch_scale') is not None:
                        new_scales = {}
                        for body in base_touch["scales"].keys():
                            # Only include body groups enabled in the config
                            if config.get(f"touch_{BODY_GROUPS.get(body, body)}") is True:
                                new_scales[body] = base_touch["scales"][body] * config['touch_scale']
                        tp["scales"] = new_scales

                    if config.get('touch_function') is not None:
                        tp["touch_function"] = config['touch_function']
                    if config.get('touch_response') is not None:
                        tp["response_function"] = config['touch_response']

                    touch_params = tp
            if config['actuation_model'] is not None:
                actuation_model = ACTUATION_MODELS[config['actuation_model']]
            
            self.behavior = config['behavior']
            self.save_dir = config['save_dir']
            self.save_logs_every = config['save_logs_every']
            self.training = kwargs['training']
        else:
            config = None
            self.behavior = None
            self.save_dir = None
            self.save_logs_every = None
            self.training = None

        super().__init__(model_path=model_path,
                         frame_skip=frame_skip,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=False,
                         done_active=False,
                         width=width,
                         height=height,)

        # Lock joints in initial position
        if config is not None:
            for joint_id in self.mimo_joints:
                joint_name = self.model.joint(joint_id).name
                body_id = self.model.joint(joint_id).bodyid
                body_name = self.model.body(body_id).name
                if config[f"lock_{BODY_GROUPS[body_name]}"] is True:
                    env_utils.lock_joint(self.model, joint_name)
        
            # Update locked joints for positional actuation model
            if isinstance(self.actuation_model, PositionalModel):
                self.actuation_model.locked_joints = self.actuation_model.get_locked_joints()
        
        # initialize 
        self.set_state(self.init_qpos, self.init_qvel)
        # perform 100 steps with no action to stabilize initial position
        self._set_action(np.zeros(self.action_space.shape))
        for _ in range(100):
            self._single_mujoco_step()
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.steps = 0

        # initialize info functions
        self._info_init()
        self._info_hist = []
                         
    def compute_reward(self, **kwargs):
        """ Dummy function for intrinsically motivated learning.
        Always returns 0.
        """
        return 0

    def reset_model(self):
        """ Resets the simulation.

        Return the simulation to the XML state, then slightly randomize all joint positions. Afterwards we let the
        simulation settle for a fixed number of steps. This leads to MIMo settling into a slightly random sitting or
        crouching position.

        Returns:
            Dict: Observations after reset.
        """

        # save info logs and reset info functions
        if self.training:
            self._info_reset()
        
        # initialize state
        self.set_state(self.init_qpos, self.init_qvel)

        # randomize initial state
        self._randomize_reset()

        self.steps = 0
        return self._get_obs()

    def _randomize_reset(self):
        """ 
        To be replaced with custom function for each behavior
        """
        pass

    def _info_reset(self):
        """
        Only called during training.
        """
        info = self._info()
        info['steps'] = self.steps
        self._info_hist.append(info)
        if (self.save_logs_every is not None) and (self.save_dir is not None):
            if self.steps % self.save_logs_every == 0:
                # Save info to file
                with open(f'{self.save_dir}/logs/training.pkl', 'wb') as f:
                    pickle.dump(self._info_hist, f, -1)
        self._info_init()

    def _step_callback(self):
        pass

    def step(self, action):
        """ Run one timestep of the environment's dynamics.

        This function takes a simulation step with the given control inputs, collects the observations, computes the
        reward and finally determines if we are done with this episode or not. :meth:`._get_obs` collects the
        observations, :meth:`.compute_reward` calculates the reward.`:meth:`._is_done` is called to determine if we
        have reached a terminal state and :meth:`._step_callback` can be used for extra functions each step, such as
        incrementing a step counter. Both the 'terminated' and 'truncated' return values are determined by
        `:meth:`._is_done`.

        Args:
            action (np.ndarray): An action provided by the agent

        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (success or failure as defined under the MDP of the task) is
                reached. In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and
                logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that
                are hidden from observations, or individual reward terms that are combined to produce the total reward.
        """
        self.do_simulation(action, self.frame_skip)
        self._step_callback()
        obs = self._get_obs()
        self._obs_callback()

        info = self._info()
        info['steps'] = self.steps
        
        terminated, truncated = self._is_done(None, self.goal, info)
        reward = self.compute_reward()
        self.steps += 1
        return obs, reward, terminated, truncated, info
        
    def _info(self):
        """
        Replace function with behavior-relevant information. Must return dict
        """
        return {}

    def _info_init(self):
        """
        Replace function with behavior-relevant information functions.
        """
        pass

    def is_success(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns ``False``.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``
        """
        return False
    
    def is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns ``False``.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``
        """
        return False

    def is_truncated(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def sample_goal(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def get_achieved_goal(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False
