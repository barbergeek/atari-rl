import random, datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from metrics import MetricLogger
from agent import AtariAgent
from wrappers import ResizeObservation, SkipFrame

env = gym.make('ALE/Frogger-v5')

# env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

#checkpoint = Path('checkpoints/2023-03-16T19-37-35/mario_net_1.chkpt')
checkpoint = Path('frogger_100k_episodes.chkpt')
game = AtariAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
game.exploration_rate = game.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state, info = env.reset()

    while True:

        #env.render()

        action = game.act(state)

        next_state, reward, truncated, terminated, info = env.step(action)
        done = truncated or terminated

        game.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=game.exploration_rate,
            step=game.curr_step
        )
