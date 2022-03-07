import numpy as np
from typing import Dict


class scoring:
    def __init__(self):
        self._tx_success = 0
        self._tx_failure = 0
        self._sensing = 0
        self._reward = 0
        self._time = 0

    def calculate_score(self, action_dict: Dict, observation_dict: Dict, num_freq_channel: int):
        observation_type = observation_dict['type']
        if observation_type == 'sensing':
            self._sensing += 1
            self._time += 1
        elif observation_type == 'tx_data_packet':
            tx_freq_channel_list = action_dict['freq_channel_list']
            success_freq_channel_list = observation_dict['success_freq_channel_list']
            failure_freq_channel_list = list(set(tx_freq_channel_list) - set(success_freq_channel_list))
            num_unit_packet = action_dict['num_unit_packet']
            tx_time = 24 * num_unit_packet
            self._time += tx_time
            self._tx_success += len(success_freq_channel_list) * tx_time
            self._tx_failure += len(failure_freq_channel_list) * tx_time
        tx_success = self._tx_success
        tx_failure = self._tx_failure
        sensing = self._sensing
        tx_success /= (self._time * num_freq_channel)
        tx_failure /= (self._time * num_freq_channel)
        sensing /= self._time
        score = tx_success - (5 * tx_failure)
        score = {'tx_success': tx_success, 'tx_failure': tx_failure, 'sensing': sensing, 'score': score}
        return score
