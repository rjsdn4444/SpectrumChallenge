from player.player import Player
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
import itertools
from collections import deque
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from player.deep_neural_network_model import DnnModel
from tensorflow.keras.optimizers import Adam


def saveCSV(result: list, path):
    path += 'result.csv'
    sc = pd.DataFrame(result, columns=['reward', 'success', 'failure', 'sensing'])
    sc.to_csv(path, index=False)


class DqnPlayer(Player):
    def __init__(self, identifier: str, max_num_unit_packet: int, observation_history_length: int,
                 sensing_unit_packet_length_ratio: int, unit_packet_success_reward: float,
                 unit_packet_failure_reward: float, dnn_layers_list: List[Dict], random_sensing_prob: float,
                 sensing_discount_factor: float, dnn_learning_rate: float, scenario: int, modelNumber: int):
        super(DqnPlayer, self).__init__(identifier)
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
        self._main_dnn: Optional[tf.keras.Model] = None
        self._target_dnn: Optional[tf.keras.Model] = None
        self._dnn_layers_list = dnn_layers_list
        self._random_sensing_prob = random_sensing_prob
        self._sensing_discount_factor = sensing_discount_factor
        self._dnn_learning_rate = dnn_learning_rate
        self._scenario = scenario
        self._modelNumber = modelNumber
        self._result = []

    def connect_to_server(self, server_address: str, server_port: int):
        super(DqnPlayer, self).connect_to_server(server_address, server_port)
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
        self._main_dnn = DnnModel(conv_layers_list=self._dnn_layers_list, num_action=self._num_action)
        self._target_dnn = DnnModel(conv_layers_list=self._dnn_layers_list, num_action=self._num_action)
        self._main_dnn.compile(optimizer=Adam(lr=self._dnn_learning_rate), loss="mse")

    def train_dnn(self, num_episodes: int, replay_memory_size: int, mini_batch_size: int, initial_epsilon: float,
                  epsilon_decay: float, min_epsilon: float, dnn_epochs: int, progress_report: bool,
                  test_run_length: int):
        epsilon = initial_epsilon
        for episode in range(num_episodes):
            if progress_report:
                print(f"Episode: {episode} (epsilon: {epsilon})")
            self.accumulate_replay_memory(replay_memory_size, epsilon, progress_report)
            observation, target_reward = self.get_mini_batch(mini_batch_size)
            checkpoint_path = "scenario_1/cp.ckpt"
            self._main_dnn.fit(observation, target_reward, epochs=dnn_epochs)
            epsilon *= epsilon_decay
            epsilon = max(epsilon, min_epsilon)
            self._target_dnn.set_weights(self._main_dnn.get_weights())
            if test_run_length > 0:
                self.test_run(test_run_length)
            # if episode % 20 == 0:
            #     self.model_save(scenario=self._scenario, modelNumber=self._modelNumber, episode=episode)
        self.model_save(scenario=self._scenario, modelNumber=self._modelNumber, episode=num_episodes, CSV=True)

    def get_main_dnn_action_and_value(self, observation: np.ndarray):
        return self.get_dnn_action_and_value(self._main_dnn, observation)

    def get_target_dnn_action_and_value(self, observation: np.ndarray):
        return self.get_dnn_action_and_value(self._target_dnn, observation)

    @staticmethod
    def get_dnn_action_and_value(dnn: tf.keras.Model, observation: np.ndarray):
        single = False
        if observation.ndim == 3:
            observation = observation[np.newaxis, ...]
            single = True
        action_value = dnn.predict(observation)
        best_action = np.argmax(action_value, axis=1)
        best_value = np.amax(action_value, axis=1)
        if single:
            best_action = best_action[0]
            best_value = best_value[0]
            action_value = action_value[0]
        return best_action, best_value, action_value

    def get_random_action(self, sensing_prob: float):
        tx_data_prob = (1 - sensing_prob) / (self._num_action - 1)
        distribution = tx_data_prob * np.ones(self._num_action)
        distribution[0] = sensing_prob
        return int(np.random.choice(np.arange(self._num_action), p=distribution))

    def get_next_action(self, observation: np.ndarray, random_prob: float):
        if np.random.rand() < random_prob:
            action = self.get_random_action(self._random_sensing_prob)
        else:
            action, _, _ = self.get_main_dnn_action_and_value(observation)
        return int(action)

    def accumulate_replay_memory(self, replay_memory_size: int, random_prob: float, progress_report: bool):
        self._replay_memory.clear()
        for ind in range(replay_memory_size):
            if progress_report:
                print(f"Replay memory sample: {ind}/{replay_memory_size}\r", end='')
            prev_observation_history = self._observation_history
            action_index = self.get_next_action(prev_observation_history, random_prob)
            action_dict = self.convert_action_index_to_dict(action_index)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict
            reward = self.get_reward(action_dict, observation_dict)
            self.update_observation_history(action_dict, observation_dict)
            current_observation_history = self._observation_history
            experience = (prev_observation_history, action_index, reward, current_observation_history)
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
            action_index, _, _ = self.get_main_dnn_action_and_value(self._observation_history)
            action_dict = self.convert_action_index_to_dict(int(action_index))
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
        observation = np.stack([x[0] for x in samples], axis=0)
        next_observation = np.stack([x[3] for x in samples], axis=0)
        _, _, target_action_reward = self.get_main_dnn_action_and_value(observation)
        _, future_reward, _ = self.get_target_dnn_action_and_value(next_observation)
        for ind, sample in enumerate(samples):
            action = sample[1]
            action_dict = self.convert_action_index_to_dict(action)
            discount_factor = self._sensing_discount_factor
            if action_dict['type'] == 'tx_data_packet':
                num_unit_packet = action_dict['num_unit_packet']
                discount_factor = discount_factor ** (num_unit_packet * self._sensing_unit_packet_length_ratio)
            immediate_reward = sample[2]
            target_action_reward[ind, action] = immediate_reward + discount_factor * future_reward[ind]
        return observation, target_action_reward

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

    def model_save(self, scenario: int, modelNumber: int, episode: int, CSV: bool = False):
        num = 1
        path = 'savedModel/scenario_%d/model_%d/' % (scenario, modelNumber)
        self._main_dnn.save(path+"episode_%d" % episode)
        if CSV:
            saveCSV(self._result, path)




