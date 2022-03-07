from arena_server.wifi_network_operator import WiFiNetworkOperator
from arena_server.mobility import WiFiRandomWaypointMobility
from arena_server.wifi_parameter import WiFiParam
from arena_server.simulator import Simulator
from arena_server.visualization import Visualization
import numpy as np
from threading import Thread
from arena_server.wifi_bot_controller import WiFiCSMAController, WiFiPeriodicController
from arena_server.remote_player_manager import RemotePlayerManager
# from arena_server.trained_model_controller import TrainedController
# from tensorflow.keras.models import load_model

num_frequency_channel = 4
freq_channel_list = np.arange(num_frequency_channel).tolist()
center_freq_list = WiFiParam.FREQUENCY_LIST_10MHZ[0: num_frequency_channel]
sim = Simulator(center_freq_list=center_freq_list)
# sim.logger.add_print_handler()
# sim.logger.add_file_handler('output.log')
# sim.logger.set_clock(clock_interval=10)
# vis = Visualization(refresh_rate=30, max_num_line_object=100, plane_size=(40, 40), node_size=0.4,
#                     activate_time_domain_view=True)
# sim.logger.add_handler(vis)

# load saved model
# model = load_model('savedModel/scenario_1/model_1')

# Network Operator 1 (Remote)
w1 = WiFiNetworkOperator(name='w1', num_sta=20, freq_channel_list=freq_channel_list)
m1 = WiFiRandomWaypointMobility(area_center=(5, 5), area_radius=15, speed_range=(30, 50),
                                avg_pause_time=3, update_interval=-1)
w1.set_mobility(m1)

# controller (trained)
# c1 = TrainedController(model=model, max_num_unit_packet=3, observation_history_length=256,
#                        sensing_unit_packet_length_ratio=24, unit_packet_success_reward=10,
#                        unit_packet_failure_reward=-50)
# w1.set_controller(c1)
sim.add_network_operator(w1)

# Network Operator 2(local)
w2 = WiFiNetworkOperator(name='w2', num_sta=20, freq_channel_list=freq_channel_list)
m2 = WiFiRandomWaypointMobility(area_center=(-5, -5), area_radius=15, speed_range=(30, 50),
                                avg_pause_time=3, update_interval=-1)
w2.set_mobility(m2)
# c2 = WiFiCSMAController(contention_window_size=16, num_unit_packet=3)
c2 = WiFiPeriodicController(num_back_off=100, num_unit_packet=1, frequency_channel_list=[0, 2])
w2.set_controller(c2)
sim.add_network_operator(w2)

# Network Operator 3(local)
w3 = WiFiNetworkOperator(name='w3', num_sta=20, freq_channel_list=freq_channel_list)
m3 = WiFiRandomWaypointMobility(area_center=(0, 0), area_radius=15, speed_range=(30, 50),
                                avg_pause_time=3, update_interval=-1)
w3.set_mobility(m3)
# c3 = WiFiCSMAController(contention_window_size=16, num_unit_packet=3)
c3 = WiFiPeriodicController(num_back_off=200, num_unit_packet=2, frequency_channel_list=[1, 3])
w3.set_controller(c3)
sim.add_network_operator(w3)

# # Network Operator 4(local)
# w4 = WiFiNetworkOperator(name='w4', num_sta=20, freq_channel_list=freq_channel_list)
# m4 = WiFiRandomWaypointMobility(area_center=(5, -5), area_radius=15, speed_range=(30, 50),
#                                 avg_pause_time=3, update_interval=-1)
# w4.set_mobility(m4)
# # c3 = WiFiCSMAController(contention_window_size=16, num_unit_packet=3)
# c4 = WiFiPeriodicController(num_back_off=150, num_unit_packet=3, frequency_channel_list=[2])
# w4.set_controller(c4)
# sim.add_network_operator(w4)
#
# # Network Operator 5(local)
# w5 = WiFiNetworkOperator(name='w5', num_sta=20, freq_channel_list=freq_channel_list)
# m5 = WiFiRandomWaypointMobility(area_center=(-5, 5), area_radius=15, speed_range=(30, 50),
#                                 avg_pause_time=3, update_interval=-1)
# w5.set_mobility(m5)
# # c3 = WiFiCSMAController(contention_window_size=16, num_unit_packet=3)
# c5 = WiFiPeriodicController(num_back_off=300, num_unit_packet=1, frequency_channel_list=[3])
# w5.set_controller(c5)
# sim.add_network_operator(w5)


# Make remote connection
remote_player_manager = RemotePlayerManager()
remote_player_manager.set_server_socket('127.0.0.1', 8000)
remote_player_manager.connect_to_players({'p1': w1})

# Start simulation
thread = Thread(target=sim.run, args=(1000000000,))
thread.start()
