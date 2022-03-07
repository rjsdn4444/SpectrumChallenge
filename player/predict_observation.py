from player.player import Player
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from collections import deque
import random
import pandas as pd
from tensorflow.keras.optimizers import Adam


def save_csv(result: list, path):
    path += 'result.csv'
    sc = pd.DataFrame(result, columns=['result_percentage', 'predict_one', 'predict_zero', 'obs_one', 'obs_zero'])
    sc.to_csv(path, index=False)


class predictObservation(Player):
    def __init__(self, identifier: str, observation_history_length: int, dnn_learning_rate: float,
                 scenario: int, modelNumber: int):
        super().__init__(identifier)
        self._freq_channel_list: List[int] = []
        self._num_freq_channel = 0
        self._freq_channel_combination = []
        self._num_freq_channel_combination = 0
        self._num_action = 0
        self._observation_history_length = observation_history_length
        self._observation_history = np.empty(0)
        self._cca_thresh = -70
        self._replay_memory = deque()
        self._observation_history_list = deque()
        self._real_output_list = deque()
        self._latest_observation_dict = None
        self._main_dnn: Optional[tf.keras.Model] = None
        self._dnn_learning_rate = dnn_learning_rate
        self._scenario = scenario
        self._modelNumber = modelNumber
        self._result = []
        self._saveCSV = 0

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=(2, 1), padding='same',
                                          kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())

        return result

    @staticmethod
    def upsample(filters, size, strides=(2, 1), apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',
                                                   kernel_initializer=initializer, use_bias=False))
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self, shape):
        inputs = tf.keras.layers.Input(shape=shape)

        down_stack = [
            self.downsample(64, (64, 1), apply_batchnorm=False),
            self.downsample(128, (64, 1)),
            self.downsample(256, (64, 1)),
            self.downsample(512, (64, 1)),
            self.downsample(512, (64, 1)),
            self.downsample(512, (64, 1)),
            self.downsample(512, (64, 1)),
            self.downsample(512, (64, 1)),
        ]

        up_stack = [
            self.upsample(512, (64, 1), apply_dropout=True),
            self.upsample(512, (64, 1), apply_dropout=True),
            self.upsample(512, (64, 1), apply_dropout=True),
            self.upsample(512, (64, 1)),
            self.upsample(256, (64, 1)),
            self.upsample(128, (64, 1)),
            self.upsample(64, (64, 1)),
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(1, (64, 1), strides=(2, 1), padding='same', kernel_initializer=initializer
                                               , activation='sigmoid')
        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
            print(x.shape)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            print(x.shape)
        # x = inputs
        # for down in down_stack:
        #     x = down(x)
        #
        # for up in up_stack:
        #     x = up(x)
        #
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def connect_to_server(self, server_address: str, server_port: int):
        super().connect_to_server(server_address, server_port)
        self._freq_channel_list = self.operator_info['freq channel list']
        self._num_freq_channel = len(self._freq_channel_list)
        self._observation_history = np.zeros((self._observation_history_length, self._num_freq_channel, 1))
        initial_action = {'type': 'sensing'}
        self._latest_observation_dict = self.step(initial_action)
        self.update_observation_history(self._latest_observation_dict)
        self._main_dnn = self.Generator(shape=[self._observation_history_length, self._num_freq_channel, 1])
        self._main_dnn.compile(optimizer=Adam(lr=self._dnn_learning_rate), loss="binary_crossentropy")

    def train_dnn(self, num_episodes: int, replay_memory_size: int, mini_batch_size: int,
                  dnn_epochs: int, progress_report: bool, test_run_length: int):
        for episode in range(num_episodes):
            if progress_report:
                print(f"Episode: {episode}")
            self.accumulate_replay_memory(replay_memory_size, progress_report)
            observation, real_output = self.get_mini_batch(mini_batch_size)
            # checkpoint_path = "scenario_1/cp.ckpt"
            self._main_dnn.fit(observation, real_output, batch_size=mini_batch_size, epochs=dnn_epochs)
            if test_run_length > 0:
                self.test_run(test_run_length)
            if episode % 20 == 0:
                self.model_save(scenario=self._scenario, modelNumber=self._modelNumber, episode=episode)
        self.model_save(scenario=self._scenario, modelNumber=self._modelNumber, episode=num_episodes, csv=True)

    def accumulate_replay_memory(self, replay_memory_size: int, progress_report: bool):
        self._replay_memory.clear()

        action_dict = {'type': 'sensing'}
        self._observation_history_list = deque(maxlen=replay_memory_size)
        self._real_output_list = deque(maxlen=replay_memory_size)
        for i in range(replay_memory_size):
            if progress_report:
                print(f"Replay memory sample: {i}/{replay_memory_size}\r", end='')
            self._observation_history_list.append(self._observation_history)
            self._real_output_list.append(self._observation_history)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict
            self.update_observation_history(observation_dict)
        for i in range(self._observation_history_length):
            self._real_output_list.append(self._observation_history)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict
            self.update_observation_history(observation_dict)
        for i in range(replay_memory_size):
            self._replay_memory.append((self._observation_history_list[i], self._real_output_list[i]))
        if progress_report:
            print()

    def test_run(self, length: int):
        for ind in range(length):
            print(f"Test run: {ind}/{length}\r", end='')
            observation_history = self._observation_history
            observation_history = np.reshape(observation_history, [1, self._observation_history_length, self._num_freq_channel, 1])
            prediction = self._main_dnn.predict(observation_history, batch_size=1)
            prediction = np.round(prediction)
            action_dict = {'type': 'sensing'}
            for i in range(self._observation_history_length):
                observation_dict = self.step(action_dict)
                self._latest_observation_dict = observation_dict
                self.update_observation_history(observation_dict)
            comparison = np.equal(self._observation_history, prediction)
            result_percentage = np.count_nonzero(comparison)/(self._observation_history_length*self._num_freq_channel)
            predict_one = np.count_nonzero(prediction == 1)
            predict_zero = np.count_nonzero(prediction == 0)
            obs_one = np.count_nonzero(observation_history == 1)
            obs_zero = np.count_nonzero(observation_history == 0)
            print(f"\n prediction and real_output match percentage:{result_percentage}"
                  f"\n prediction count 1: {predict_one}   0: {predict_zero} observation count 1: {obs_one}   0: {obs_zero}")

            self._result.append([result_percentage, predict_one, predict_zero, obs_one, obs_zero])

    def get_mini_batch(self, batch_size: int):
        samples = random.sample(self._replay_memory, batch_size)
        observation = np.stack([x[0] for x in samples], axis=0)
        real_output = np.stack([x[1] for x in samples], axis=0)
        return observation, real_output

    def update_observation_history(self, observation: Dict):
        observation_type = observation['type']
        new_observation = np.zeros((self._num_freq_channel, 1))
        new_observation_length = 1
        if observation_type == 'sensing':
            sensed_power = observation['sensed_power']
            occupied_channel_list = [int(freq_channel) for freq_channel in sensed_power
                                     if sensed_power[freq_channel] > self._cca_thresh]
            new_observation[occupied_channel_list] = 1
            new_observation_length = 1

        new_observation = np.broadcast_to(new_observation, (new_observation_length, self._num_freq_channel, 1))
        self._observation_history = np.concatenate((new_observation, self._observation_history),
                                                   axis=0)[:self._observation_history_length, ...]

    def model_save(self, scenario: int, modelNumber: int, episode: int, csv: bool = False):
        num = 1
        path = 'savedModel/scenario_%d/model_%d/' % (scenario, modelNumber)
        self._main_dnn.save(path+"episode_%d" % episode)
        if csv:
            save_csv(self._result, path)
