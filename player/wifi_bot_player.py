from player.player import Player
import pandas as pd
import numpy as np


class WiFiCSMAPlayer(Player):
    def __init__(self, identifier, contention_window_size, num_unit_packet, sensing_unit_packet_length_ratio):
        super(WiFiCSMAPlayer, self).__init__(identifier)
        self._contention_window_size = contention_window_size
        self._min_contention_window_size = contention_window_size
        self._max_contention_window_size = 66536
        self._num_unit_packet = num_unit_packet
        self._sensing_unit_packet_length_ratio = sensing_unit_packet_length_ratio
        self._freq_channel_list = []
        self._num_freq_channel = 0
        self._primary_channel = 0
        self._back_off: int = np.random.randint(1, self._contention_window_size)
        self._cca_thresh = -70
        self._result = []

    def run(self, execution_number, count):
        action = {'type': 'sensing'}
        self._freq_channel_list = self.operator_info['freq channel list']
        self._primary_channel = self._freq_channel_list[0]
        self._num_freq_channel = len(self._freq_channel_list)
        for i in range(count):
            tx_success = 0
            tx_failure = 0
            sensing = 0
            time = 0
            for it in range(execution_number):
                observation = self.step(action)
                observation_type = observation['type']
                if observation_type == 'sensing':
                    sensing += 1
                    time += 1
                    sensed_power = observation['sensed_power']
                    sensed_power = {int(freq_channel): sensed_power[freq_channel] for freq_channel in sensed_power}
                    primary_sensed_power = sensed_power[self._primary_channel]
                    if primary_sensed_power > self._cca_thresh:
                        action = {'type': 'sensing'}
                    else:
                        self._back_off -= 1
                        if self._back_off <= 0:
                            freq_channel_list = []
                            for ch in sensed_power:
                                if sensed_power[ch] <= self._cca_thresh:
                                    freq_channel_list.append(ch)
                            action = {'type': 'tx_data_packet', 'freq_channel_list': freq_channel_list,
                                      'num_unit_packet': self._num_unit_packet}
                        else:
                            action = {'type': 'sensing'}
                elif observation_type == 'tx_data_packet':
                    tx_freq_channel_list = action['freq_channel_list']
                    success_freq_channel_list = observation['success_freq_channel_list']
                    failure_freq_channel_list = list(set(tx_freq_channel_list) - set(success_freq_channel_list))
                    num_unit_packet = action['num_unit_packet']
                    tx_time = self._sensing_unit_packet_length_ratio * num_unit_packet
                    time += tx_time
                    tx_success += len(success_freq_channel_list) * tx_time
                    tx_failure += len(failure_freq_channel_list) * tx_time
                    if self._primary_channel in success_freq_channel_list:
                        self._contention_window_size = self._min_contention_window_size
                    else:
                        self._contention_window_size = min(self._contention_window_size*2,
                                                           self._max_contention_window_size)
                    self._back_off = np.random.randint(1, self._contention_window_size)
                    action = {'type': 'sensing'}
            tx_success /= (time * self._num_freq_channel)
            tx_failure /= (time * self._num_freq_channel)
            sensing /= time
            self._result.append([tx_success, tx_failure, sensing])
            print(f"\n Sensing: {sensing}, Tx Success: {tx_success}, Tx Failure: {tx_failure}")
        path = 'savedModel/'
        self.saveCSV(self._result, path)

    @staticmethod
    def saveCSV(result: list, path):
        path += 'CSMAresult_CSMA.csv'
        sc = pd.DataFrame(result, columns=['success', 'failure', 'sensing'])
        sc.to_csv(path, index=False)