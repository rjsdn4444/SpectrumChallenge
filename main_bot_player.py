from player.wifi_bot_player import WiFiCSMAPlayer


identifier = 'p1'
contention_window_size = 16
player = WiFiCSMAPlayer(identifier=identifier, contention_window_size=contention_window_size, num_unit_packet=2,
                        sensing_unit_packet_length_ratio=24)
player.connect_to_server('127.0.0.1', 8000)
player.run(execution_number=1000, count=100)
