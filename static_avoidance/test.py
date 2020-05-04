import logging
import argparse
import configparser
import os
import torch
import numpy as np
from explorer import Explorer
from utils.robot import Robot
from policy import SOA
from env import Env
from memory import ReplayMemory


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='SOA')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model2.pth')):
            model_weights = os.path.join(args.model_dir, 'resumed_rl_model2.pth')
        #else:
            #model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = SOA()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    policy.get_model().load_state_dict(torch.load(model_weights))
    model = policy.get_model()

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = Env()
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation

    policy.set_env(env)
    robot.print_info()


    #explorer = Explorer(env, robot, device, memory=memory, gamma=0.9, target_policy=policy)
    #explorer.update_target_model(model)
    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob)
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))

    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
        #explorer.run_k_episodes(1, 'train', update_memory=True, print_failure=True)


if __name__ == '__main__':
    main()