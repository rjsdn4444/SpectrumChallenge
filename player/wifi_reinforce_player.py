from player.player import Player
from typing import Dict, List, Optional
import numpy as np
import torch
import itertools
from collections import deque
import random
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from player.policy_network_model_torch import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import torch.nn as nn
from torchsummary import summary
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


def saveCSV(result: list, path):
    path += 'result.csv'
    sc = pd.DataFrame(result, columns=['reward', 'success', 'failure', 'sensing'])
    sc.to_csv(path, index=False)


class PPOPlayer(Player):
    def __init__(self, identifier: str, max_num_unit_packet: int, observation_history_length: int,
                 sensing_unit_packet_length_ratio: int, unit_packet_success_reward: float,
                 unit_packet_failure_reward: float, dnn_layers_list: List[Dict], sensing_discount_factor: float,
                 dnn_learning_rate: float, scenario: int, modelNumber: int, clipping_val: float, entropy_beta: float,
                 lmbda: float, value_function_coef: float):
        super(PPOPlayer, self).__init__(identifier)
        self._freq_channel_list: List[int] = []
        self._num_freq_channel = 0
        self._max_num_unit_packet = max_num_unit_packet
        self._freq_channel_combination = []
        self._num_freq_channel_combination = 0
        self._num_action = 0
        self._observation_history_length = observation_history_length
        self._sensing_unit_packet_length_ratio = sensing_unit_packet_length_ratio
        self._observation_history = np.empty(0)
        self._cca_thresh = -70
        self._replay_memory = deque()
        self._latest_observation_dict = None
        self._unit_packet_success_reward = unit_packet_success_reward
        self._unit_packet_failure_reward = unit_packet_failure_reward
        self._device = None
        self._main_dnn: Optional[torch.nn.Module] = None
        self._model_optimizer: Optional[torch.optim] = None
        self._mse_loss = nn.MSELoss
        self._dnn_layers_list = dnn_layers_list
        self._sensing_discount_factor = sensing_discount_factor
        self._dnn_learning_rate = dnn_learning_rate
        self._scenario = scenario
        self._modelNumber = modelNumber
        self._result = []
        self._lmbda = lmbda
        self._clipping_val = clipping_val
        self._entropy_beta = entropy_beta
        self._value_function_coef = value_function_coef

    def connect_to_server(self, server_address: str, server_port: int):
        super(PPOPlayer, self).connect_to_server(server_address, server_port)
        self._freq_channel_list = self.operator_info['freq channel list']
        self._num_freq_channel = len(self._freq_channel_list)
        self._freq_channel_combination = [np.where(np.flip(np.array(x)))[0].tolist()
                                          for x in itertools.product((0, 1), repeat=self._num_freq_channel)][1:]
        self._num_freq_channel_combination = 2 ** self._num_freq_channel - 1
        self._num_action = self._num_freq_channel_combination * self._max_num_unit_packet + 1
        self._observation_history = np.zeros((self._observation_history_length, self._num_freq_channel, 2))
        initial_action = {'type': 'sensing'}
        self._latest_observation_dict = self.step(initial_action)
        self.update_observation_history(initial_action, self._latest_observation_dict)
        self._device = torch.device('cpu')
        if torch.cuda.is_available():
            self._device = torch.device('cuda:0')
            torch.cuda.empty_cache()
        self._main_dnn = ResNet18(self._num_action).to(self._device)
        self._model_optimizer = torch.optim.Adam([{'params': self._main_dnn.parameters(),
                                                   'lr': self._dnn_learning_rate}])

    def train_dnn(self, num_episodes: int, replay_memory_size: int, mini_batch_size: int, dnn_epochs: int,
                  progress_report: bool, test_run_length: int):
        for episode in range(num_episodes):
            if progress_report:
                print(f"Episode: {episode}")
            self.accumulate_replay_memory(replay_memory_size, progress_report)
            _, discounted_rewards, actions, observations = self.get_mini_batch(mini_batch_size)
            observations = torch.squeeze(observations).detach().to(self._device)
            actions = torch.squeeze(actions).detach().to(self._device)
            for epoch in range(dnn_epochs):
                action_logprobs, _ = self.get_evaluation_action(observations, actions)
                total_loss = -action_logprobs * discounted_rewards
                total_loss = total_loss.mean()
                self._model_optimizer.zero_grad()
                total_loss.backward()
                self._model_optimizer.step()
            if test_run_length > 0:
                self.test_run(test_run_length)
        self.model_save(scenario=self._scenario, modelNumber=self._modelNumber)

    def get_evaluation_action(self, observation: np.ndarray, action: int):
        """
        For updating deep neural network without detach
        For calculating loss functions
        :param observation: observation in trajectories
        :param action: actions in trajectories
        :return: log probabilities of actions, values by critic network, entropy of distribution
        """
        action_probs = self._main_dnn(observation)
        distribution = Categorical(logits=action_probs)
        action_logprobs = distribution.log_prob(action)
        distribution_entropy = distribution.entropy()
        return action_logprobs, distribution_entropy

    def get_next_action(self, observation: np.ndarray):
        """
        For stacking trajectories with policy
        there is no update (detach)
        :param observation: observation from environment
        :return: action(index), action's log probability
        """
        observation = observation[np.newaxis, ...]
        # observation = torch.Tensor(observation)
        action_probs = self._main_dnn(observation)
        distribution = Categorical(logits=action_probs)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)
        return action, action_logprob

    def accumulate_replay_memory(self, replay_memory_size: int, progress_report: bool):
        self._replay_memory.clear()
        for ind in range(replay_memory_size):
            if progress_report:
                print(f"Replay memory sample: {ind}/{replay_memory_size}\r", end='')
            prev_observation_history = self._observation_history
            prev_observation_history = prev_observation_history.transpose((2, 0, 1))
            with torch.no_grad():
                prev_observation_history = torch.FloatTensor(prev_observation_history).to(self._device)
                action, action_logprob = self.get_next_action(prev_observation_history)
            action_index = int(action)
            action_dict = self.convert_action_index_to_dict(action_index)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict
            reward = self.get_reward(action_dict, observation_dict)
            # print(reward)
            self.update_observation_history(action_dict, observation_dict)
            experience = (action, reward, action_logprob, prev_observation_history)
            self._replay_memory.append(experience)
        if progress_report:
            print()

    def test_run(self, length: int):
        tx_success = 0
        tx_failure = 0
        sensing = 0
        reward = 0
        time = 0
        for ind in range(length):
            print(f"Test run: {ind}/{length}\r", end='')
            prev_observation_history = self._observation_history
            prev_observation_history = prev_observation_history.transpose((2, 0, 1))
            with torch.no_grad():
                prev_observation_history = torch.FloatTensor(prev_observation_history).to(self._device)
                action, _ = self.get_next_action(prev_observation_history)
            action_dict = self.convert_action_index_to_dict(int(action))
            observation_dict = self.step(action_dict)
            self.update_observation_history(action_dict, observation_dict)
            observation_type = observation_dict['type']
            if observation_type == 'sensing':
                sensing += 1
                time += 1
            elif observation_type == 'tx_data_packet':
                tx_freq_channel_list = action_dict['freq_channel_list']
                success_freq_channel_list = observation_dict['success_freq_channel_list']
                failure_freq_channel_list = list(set(tx_freq_channel_list) - set(success_freq_channel_list))
                num_unit_packet = action_dict['num_unit_packet']
                tx_time = self._sensing_unit_packet_length_ratio * num_unit_packet
                time += tx_time
                tx_success += len(success_freq_channel_list) * tx_time
                tx_failure += len(failure_freq_channel_list) * tx_time
            reward += self.get_reward(action_dict, observation_dict)
        reward /= time
        tx_success /= (time * self._num_freq_channel)
        tx_failure /= (time * self._num_freq_channel)
        sensing /= time
        self._result.append([reward, tx_success, tx_failure, sensing])
        print(f"\nReward: {reward}, Sensing: {sensing}, Tx Success: {tx_success}, Tx Failure: {tx_failure}")

    def get_mini_batch(self, batch_size: int):
        samples = random.sample(self._replay_memory, batch_size)
        actions = torch.stack([x[0] for x in samples], dim=0)
        rewards = np.stack([x[1] for x in samples], axis=0)
        logprobs = torch.stack([x[2] for x in samples], dim=0)
        observations = torch.stack([x[3] for x in samples], dim=0)
        discount_factors = []
        for ind, sample in enumerate(samples):
            action = sample[0]
            action_dict = self.convert_action_index_to_dict(action)
            discount_factor = self._sensing_discount_factor
            if action_dict['type'] == 'tx_data_packet':
                num_unit_packet = action_dict['num_unit_packet']
                discount_factor = discount_factor ** (num_unit_packet * self._sensing_unit_packet_length_ratio)
            discount_factors.append(discount_factor)
        discounted_rewards = self.get_discounted_rewards(rewards, discount_factors)
        return logprobs, discounted_rewards, actions, observations

    def get_discounted_rewards(self, rewards, discount_factors):
        discounted_rewards = []
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            discounted_reward = rewards[i] + (discount_factors[i] * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self._device)
        eps = np.finfo(np.float32).eps.item()
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + eps)
        return discounted_rewards

    def convert_action_index_to_dict(self, action_index: int) -> Dict:
        """ Convert action index to dictionary form
        Args:
            action_index: index of action (0: sensing, 1 to (2^num_freq_channel-1)*max_num_unit_packet: tx_data_packet)
        Returns:
            action in dictionary form
                'type': 'sensing' or 'tx_data_packet',
                'freq_channel_list': list of frequency channels for data transmission
                'num_unit_packet': number of unit packets
        """
        if action_index == 0:
            action_dict = {'type': 'sensing'}
        else:
            num_unit_packet = (action_index - 1) // self._num_freq_channel_combination + 1
            # print(num_unit_packet)
            freq_channel_combination_index = (action_index - 1) % self._num_freq_channel_combination
            freq_channel_list = self._freq_channel_combination[freq_channel_combination_index]
            action_dict = {'type': 'tx_data_packet', 'freq_channel_list': freq_channel_list,
                           'num_unit_packet': num_unit_packet}
        return action_dict

    def update_observation_history(self, action: Dict, observation: Dict):
        observation_type = observation['type']
        new_observation = np.zeros((self._num_freq_channel, 2))
        new_observation_length = 1
        if observation_type == 'sensing':
            sensed_power = observation['sensed_power']
            occupied_channel_list = [int(freq_channel) for freq_channel in sensed_power
                                     if sensed_power[freq_channel] > self._cca_thresh]
            new_observation[occupied_channel_list, 0] = 1
            new_observation_length = 1
        elif observation_type == 'tx_data_packet':
            tx_freq_channel_list = action['freq_channel_list']
            success_freq_channel_list = observation['success_freq_channel_list']
            failure_freq_channel_list = list(set(tx_freq_channel_list) - set(success_freq_channel_list))
            num_unit_packet = action['num_unit_packet']
            new_observation[failure_freq_channel_list, 1] = 1

            new_observation_length = num_unit_packet * self._sensing_unit_packet_length_ratio
        new_observation = np.broadcast_to(new_observation, (new_observation_length, self._num_freq_channel, 2))
        self._observation_history = np.concatenate((new_observation, self._observation_history),
                                                   axis=0)[:self._observation_history_length, ...]

    def get_reward(self, action: Dict, observation: Dict):
        observation_type = observation['type']
        reward = 0
        if observation_type == 'sensing':
            reward = 0
        elif observation_type == 'tx_data_packet':
            num_unit_packet = action['num_unit_packet']
            num_tx_packet = len(action['freq_channel_list'])
            num_success_packet = len(observation['success_freq_channel_list'])
            num_failure_packet = num_tx_packet - num_success_packet
            reward = (num_success_packet * self._unit_packet_success_reward +
                      num_failure_packet * self._unit_packet_failure_reward) * num_unit_packet
        return reward

    def model_save(self, scenario: int, modelNumber: int):
        num = 1
        scenario = scenario
        path = 'savedModel/scenario_%d/model_%d' % (scenario, modelNumber)
        # while os.path.exists(path):
        #     num += 1
        #     path = 'savedModel/scenario_%d/model_%d' % (scenario, modelNumber)
        torch.save(self._main_dnn.state_dict(), path + '.pt')
        saveCSV(self._result, path)




