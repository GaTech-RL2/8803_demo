import time

import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as trans


def test_osc_control_delta():
    from custom_env import Custom_Lift

    options = {}
    options["robots"] = ["Panda"]
    controller_conf = suite.controllers.load_controller_config(default_controller="OSC_POSE")
    controller_conf["control_delta"] = True
    options["controller_configs"] = controller_conf
    options["env_name"] = "Custom_Lift"

    view_name = "frontview"
    options["camera_names"] = view_name
    options["render_camera"] = view_name

    env = suite.make(**options,
                     has_renderer=True,
                     has_offscreen_renderer=False,
                     ignore_done=True,
                     use_camera_obs=False,
                     horizon=200,
                     control_freq=20,
                     use_object_obs=True)

    obs = env.reset()
    # reset the environment
    print(obs.keys())
    for k in obs:
        print(k, obs[k].shape)

    for i in range(300):
        # action = np.random.randn(7, )  # sample random action
        action = np.array([0, 0, -0.05, 0, 0, 0, -1])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        print("end-effector position: ", obs["robot0_eef_pos"])



test_osc_control_delta()