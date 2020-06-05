# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:26:49 2020

@author: Hill
"""
import gym
import torch
from algorithm import SAC
import matplotlib.pyplot as plt
#from arguments import get_args

def main(env, agent, Episode, batch_size):
    Return = []
    for episode in range(Episode):
        score = 0
        state = env.reset()
        for i in range(300):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, next_state, done_mask))
            state = next_state

            score += reward
            if done:
                break
            if agent.buffer.buffer_len() > 500:
                agent.update(batch_size)

        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent.buffer.buffer_len()))
        Return.append(score)
        score = 0
    env.close()
    plt.plot(Return)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Params
    tau = 0.01
    gamma = 0.99
    q_lr = 3e-3
    value_lr = 3e-3
    policy_lr = 3e-3
    buffer_maxlen = 50000

    Episode = 100
    batch_size = 128

    agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr)
    main(env, agent, Episode, batch_size)