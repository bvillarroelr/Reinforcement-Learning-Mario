# src/env_setup.py
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def make_mario_env():
    # 1. Cargar ROM
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # 2. Controles simples
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # 3. Escala de grises
    env = GrayScaleObservation(env, keep_dim=True)
    # 4. Redimensionar a 84x84
    env = ResizeObservation(env, shape=84)
    # 5. Wrapper necesario para Stable Baselines
    env = DummyVecEnv([lambda: env])
    # 6. Apilar 4 frames
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env