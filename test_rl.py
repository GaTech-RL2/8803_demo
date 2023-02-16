from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
import robosuite as suite
from robosuite.wrappers import GymWrapper

filename = str(uuid.uuid4())

def setup_env():
    options = {}
    options["robots"] = ["Panda"]
    controller_conf = suite.controllers.load_controller_config(default_controller="OSC_POSE")
    controller_conf["control_delta"] = True
    options["controller_configs"] = controller_conf
    options["env_name"] = "Lift"

    view_name = "frontview"
    options["camera_names"] = view_name
    options["render_camera"] = view_name

    env = suite.make(**options,
                     has_renderer=True,
                     has_offscreen_renderer=False,
                     ignore_done=False,
                     use_camera_obs=False,
                     horizon=200,
                     control_freq=20,
                     use_object_obs=True)

    env = GymWrapper(env, keys=["robot0_proprio-state", "object-state"])
    return env

def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    env = setup_env()
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="./local/epoch_0_params.pkl",
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=200,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
