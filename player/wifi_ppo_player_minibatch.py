from player.player import Player
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
import itertools
from collections import deque
import random
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from player.actor_critic_nn_model import DnnModel
from player.actor_critic_nn_model import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow_probability as tfp


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
        self._main_dnn: Optional[tf.keras.Model] = None
        self._model_optimizer: Optional[tf.keras.optimizers] = None
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
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._log_dir = 'logs\\gradient_tape\\' + current_time + '\\loss'
        self._result_log_dir = 'logs\\gradient_tape\\' + current_time + '\\result'
        self._summary_writer = tf.summary.create_file_writer(self._log_dir)
        self._result_summary_writer = tf.summary.create_file_writer(self._result_log_dir)
        self._episode = 0

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
        self._main_dnn = resnet_18(num_action=self._num_action, writer=self._summary_writer, step=self._episode)
        self._model_optimizer = SGD(lr=self._dnn_learning_rate)

    def train_dnn(self, num_episodes: int, replay_memory_size: int, mini_batch_size: int, dnn_epochs: int,
                  progress_report: bool, test_run_length: int):
        for episode in range(num_episodes):
            if progress_report:
                print(f"Episode: {episode}")
            self.accumulate_replay_memory(replay_memory_size, progress_report)
            observation, returns, advantages, old_policies, old_values = self.get_mini_batch(mini_batch_size)
            returns = tf.reshape(returns, (len(returns),))
            advantages = tf.reshape(advantages, (len(advantages),))
            old_policies = tf.reshape(old_policies, (len(old_policies), self._num_action))
            old_values = tf.reshape(old_values, (len(old_values), 1))
            for epoch in range(dnn_epochs):
                with tf.GradientTape() as tape1:
                    policy, value = self._main_dnn(observation, training=True)
                    value = tf.reshape(value, (len(value),))
                    critic_loss = self.critic_loss(value, old_values, returns)
                    actor_loss = self.actor_loss(policy, advantages, old_policies, critic_loss)
                gradient = tape1.gradient(actor_loss, self._main_dnn.trainable_variables)
                self._model_optimizer.apply_gradients(zip(gradient, self._main_dnn.trainable_variables))
                with self._summary_writer.as_default():
                    tf.summary.scalar('loss', actor_loss, step=epoch)
            if test_run_length > 0:
                self.test_run(test_run_length)
            self._episode += 1
        self.model_save(scenario=self._scenario, modelNumber=self._modelNumber, episode=num_episodes, CSV=True)

    def actor_loss(self, probabilities, advantages, oldpolicy_probs, critic_loss):
        probability = probabilities
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))
        general_loss = []
        clipped_loss = []
        for prob, adv, old_prob in zip(probability, advantages, oldpolicy_probs):
            adv = - tf.constant(adv)
            old_prob = tf.constant(old_prob)
            ratio = tf.math.divide(prob, old_prob)
            general_loss.append(tf.math.multiply(ratio, adv))
            clipped_loss.append(tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self._clipping_val,
                                                                  1.0 + self._clipping_val), adv))
        stack_general_loss = tf.stack(general_loss)
        stack_clipped_loss = tf.stack(clipped_loss)
        loss = tf.reduce_mean(tf.math.maximum(stack_general_loss, stack_clipped_loss)) +\
               self._value_function_coef * critic_loss - self._entropy_beta * entropy
        return loss

    def critic_loss(self, values, old_values, returns):
        general_loss = []
        clipped_loss = []
        for value, old_value, ret in zip(values, old_values, returns):
            value = tf.constant(value)
            old_value = tf.constant(old_value)
            value_clipped = old_value + tf.clip_by_value(value - old_value, -self._clipping_val, self._clipping_val)
            general_loss.append(tf.square(value - ret))
            clipped_loss.append(tf.square(value_clipped - ret))
        stack_general_loss = tf.stack(general_loss)
        stack_clipped_loss = tf.stack(clipped_loss)
        critic_loss = 0.5 * tf.reduce_mean(tf.maximum(stack_general_loss, stack_clipped_loss))
        return critic_loss

    def get_random_action(self, sensing_prob: float):
        tx_data_prob = (1 - sensing_prob) / (self._num_action - 1)
        distribution = tx_data_prob * np.ones(self._num_action)
        distribution[0] = sensing_prob
        return int(np.random.choice(np.arange(self._num_action), p=distribution))

    def get_next_action(self, observation: np.ndarray):
        action_prob, value = self._main_dnn(np.array([observation]))
        action_prob = action_prob.numpy()
        distribution = tfp.distributions.Categorical(probs=action_prob, dtype=tf.float32)
        action = distribution.sample()
        # print(value)
        return int(action.numpy()[0]), action_prob, value

    def accumulate_replay_memory(self, replay_memory_size: int, progress_report: bool):
        self._replay_memory.clear()
        for ind in range(replay_memory_size):
            if progress_report:
                print(f"Replay memory sample: {ind}/{replay_memory_size}\r", end='')
            prev_observation_history = self._observation_history
            action_index, action_prob, old_value = self.get_next_action(prev_observation_history)
            action_dict = self.convert_action_index_to_dict(action_index)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict
            reward = self.get_reward(action_dict, observation_dict)
            self.update_observation_history(action_dict, observation_dict)
            current_observation_history = self._observation_history
            _, value = self._main_dnn(np.array([current_observation_history]))
            experience = (prev_observation_history, action_index, reward, action_prob, old_value, value)
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
            action_index, _, _ = self.get_next_action(self._observation_history)
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
        with self._result_summary_writer.as_default():
            tf.summary.scalar('reward', reward, step=self._episode)
            tf.summary.scalar('Tx Success', tx_success, step=self._episode)
            tf.summary.scalar('Tx, Failure', tx_failure, step=self._episode)
            tf.summary.scalar('Sensing', sensing, step=self._episode)

    def get_mini_batch(self, batch_size: int):
        samples = random.sample(self._replay_memory, batch_size)
        observation = np.stack([x[0] for x in samples], axis=0)
        rewards = np.stack([x[2] for x in samples], axis=0)
        old_policies = np.stack([x[3] for x in samples], axis=0)
        old_values = np.stack([x[4] for x in samples], axis=0)
        values = np.stack([x[5] for x in samples], axis=0)
        discount_factors = []
        for ind, sample in enumerate(samples):
            action = sample[1]
            action_dict = self.convert_action_index_to_dict(action)
            discount_factor = self._sensing_discount_factor
            if action_dict['type'] == 'tx_data_packet':
                num_unit_packet = action_dict['num_unit_packet']
                discount_factor = discount_factor ** (num_unit_packet * self._sensing_unit_packet_length_ratio)
            discount_factors.append(discount_factor)
        returns, advantages = self.get_advantages(old_values, values, rewards, discount_factors)
        return observation, returns, advantages, old_policies, old_values

    def get_advantages(self, old_values, values, rewards, discount_factors):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + discount_factors[i] * values[i] - old_values[i]
            gae = delta + discount_factors[i] * self._lmbda * gae
            returns.insert(0, gae + values[i])
        adv = np.array(returns, dtype=np.float32) - old_values
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        returns = np.array(returns, dtype=np.float32)
        return returns, adv

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




