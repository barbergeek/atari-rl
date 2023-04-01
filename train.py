import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from metrics import MetricLogger
from agent import AtariAgent
from wrappers import ResizeObservation, SkipFrame

startTime = datetime.datetime.now()

# Initialize game environment
env = gym.make('ALE/Frogger-v5')

# Apply Wrappers to environment
#env = SkipFrame(env, skip=4) # ALE automatically skips frames, don't do it again...
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('frogger_50k_episodes.chkpt')
game = AtariAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 50000
#episodes = 100

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state, info = env.reset()

    # Play the game!
    while True:

        # 4. Run agent on the state
        action = game.act(state)

        # 5. Agent performs action
        next_state, reward, truncated, terminated, info = env.step(action)
        done = truncated or terminated

        # 6. Remember
        game.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = game.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done:
            break

    logger.log_episode()

    if e % 50 == 0:
        logger.record(
            episode=e,
            epsilon=game.exploration_rate,
            step=game.curr_step
        )

# save the final model
game.save()

delta = datetime.datetime.now() - startTime
print("Elapsed time = {}".format(delta))