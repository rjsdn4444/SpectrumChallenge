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
from player.actor_critic_nn_model import Actor, Critic, DnnModel
from utility.ppo_loss import ppo_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls


def saveCSV(result: list, path):
    path += 'result.csv'
    sc = pd.DataFrame(result, columns=['reward', 'success', 'failure', 'sensing'])
    sc.to_csv(path, index=False)


class PPOPlayer(Player):
    def __init__(self, identifier: str, max_num_unit_packet: int, observation_history_length: int,
                 sensing_unit_packet_length_ratio: int, unit_packet_success_reward: float,
                 unit_packet_failure_reward: float, dnn_layers_list: List[Dict], dnn_learning_rate: float,
                 clipping_val: float, critic_discount: float, entropy_beta: float, gamma: float, lmbda: float,
                 scenario: int, model_number: int):
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
        self._latest_observation_dict = None
        self._unit_packet_success_reward = unit_packet_success_reward
        self._unit_packet_failure_reward = unit_packet_failure_reward
        self._dnn_model: Optional[tf.keras.Model] = None
        # self._critic: Optional[tf.keras.Model] = None
        self._model_optimizer: Optional[tf.keras.optimizers] = None
        self._critic_optimizer: Optional[tf.keras.optimizers] = None
        self._dnn_layers_list = dnn_layers_list
        self._dnn_learning_rate = dnn_learning_rate
        self._scenario = scenario
        self._modelNumber = model_number
        self._result = []
        self._gamma = gamma
        self._lmbda = lmbda
        self._clipping_val = clipping_val
        self._critic_discount = critic_discount
        self._entropy_beta = entropy_beta
        self._observation_samples, self._action_smaples, self._q_value_samples = [], [], []
        self._reward_samples, self._actions_prob_samples = [], []
        self._value_function_coef = 0.5

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
        self._dnn_model = DnnModel(self._dnn_layers_list, self._num_action)
        # self._critic = Critic(self._dnn_layers_list, 1)
        self._model_optimizer = Adam(lr=self._dnn_learning_rate)
        # self._critic_optimizer = Adam(lr=self._dnn_learning_rate)

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

    def train_dnn(self, num_episodes: int, dnn_epochs: int, progress_report: bool, replay_memory_size: int,
                  test_run_length: int):
        for episode in range(num_episodes):
            if progress_report:
                print(f"Episode: {episode}")
            self.accumulate_replay_memory(replay_memory_size, progress_report)
            np.reshape(self._actions_prob_samples, (len(self._actions_prob_samples), self._num_action))
            self._actions_prob_samples = np.stack(self._actions_prob_samples, axis=0)
            returns, advantages = self.get_advantages(self._q_value_samples, self._reward_samples)
            returns = np.array(returns, dtype=np.float32)
            for epoch in range(dnn_epochs):
                returns = tf.reshape(returns, (len(returns),))
                advantages = tf.reshape(advantages, (len(advantages),))
                old_policies = self._actions_prob_samples
                old_policies = tf.reshape(old_policies, (len(old_policies), self._num_action))
                old_values = self._q_value_samples
                old_values = tf.reshape(old_values, (len(old_values), 1))
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    policy, value = self._dnn_model(np.array(self._observation_samples, dtype=np.float32), training=True)
                    # value = self._critic(np.array(self._observation_samples, dtype=np.float32), training=True)
                    value = tf.reshape(value, (len(value),))
                    critic_loss = self.critic_loss(value, old_values, returns)
                    actor_loss = self.actor_loss(policy, advantages, old_policies, critic_loss)
                gradient = tape1.gradient(actor_loss, self._dnn_model.trainable_variables)
                # critic_gradient = tape2.gradient(critic_loss, self._critic.trainable_variables)
                self._model_optimizer.apply_gradients(zip(gradient, self._dnn_model.trainable_variables))
                # self._critic_optimizer.apply_gradients(zip(critic_gradient, self._critic.trainable_variables))
                # print('actor_loss: ' + actor_loss + ', critic_loss: ' + critic_loss)
            if test_run_length > 0:
                self.test_run(test_run_length)
        self.model_save(scenario=self._scenario, modelNumber=self._modelNumber, episode=num_episodes, CSV=True)

    def get_next_action(self, observation: np.ndarray):
        action_prob, _ = self._dnn_model(np.array([observation]))
        action_prob = action_prob.numpy()
        distribution = tfp.distributions.Categorical(probs=action_prob, dtype=tf.float32)
        action = distribution.sample()
        return int(action.numpy()[0]), action_prob

    def accumulate_replay_memory(self, replay_memory_size: int, progress_report: bool):
        self._observation_samples, self._action_smaples, self._q_value_samples = [], [], []
        self._reward_samples, self._actions_prob_samples = [], []
        for ind in range(replay_memory_size):
            if progress_report:
                print(f"Replay memory sample: {ind}/{replay_memory_size}\r", end='')
            prev_observation_history = self._observation_history
            self._observation_samples.append(prev_observation_history)
            action_index, action_prob = self.get_next_action(prev_observation_history)
            self._actions_prob_samples.append(action_prob[0])

            _, q_value = self._dnn_model(np.array([prev_observation_history]))
            q_value = q_value.numpy()
            self._q_value_samples.append(q_value[0][0])

            self._action_smaples.append(action_index)

            action_dict = self.convert_action_index_to_dict(action_index)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict

            reward = self.get_reward(action_dict, observation_dict)
            self._reward_samples.append(reward)

            self.update_observation_history(action_dict, observation_dict)
        _, q_value = self._dnn_model(np.array([self._observation_history]))
        q_value = q_value.numpy()
        self._q_value_samples.append(q_value[0][0])
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
            action_index, _ = self.get_next_action(self._observation_history)
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

    # def ppo_loss(self, oldpolicy_probs, advantages, rewards, values):
    #     clipping_val = self._clipping_val
    #     critic_discount = self._critic_discount
    #     entropy_beta = self._entropy_beta
    #
    #     def loss(y_true, y_pred):
    #         newpolicy_probs = y_pred
    #         ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
    #         CPI_loss = ratio * advantages
    #         clipped_loss = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
    #         actor_loss = -K.mean(K.minimum(CPI_loss, clipped_loss))
    #         critic_loss = K.mean(K.square(rewards - values))
    #         total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
    #             -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
    #         return total_loss
    #
    #     return loss

    def get_advantages(self, values, rewards):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self._gamma * values[i + 1] - values[i]
            gae = delta + self._gamma * self._lmbda * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns, dtype=np.float32) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def model_save(self, scenario: int, modelNumber: int, episode: int, CSV: bool = False):
        num = 1
        path = 'savedModel/scenario_%d/model_%d/' % (scenario, modelNumber)
        self._dnn_model.save(path+"actor_episode_%d" % episode)
        # self._critic.save(path + "critic_episode_%d" % episode)
        if CSV:
            saveCSV(self._result, path)




