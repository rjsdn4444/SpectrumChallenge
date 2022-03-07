from player.wifi_a2c_player_torch import PPOPlayer


dnn_layers_list = [
    {'filters': 32,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_norm_size': 32,
     'max_pool_size': (2, 1)},
    {'filters': 32,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_norm_size': 64,
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_norm_size': 64,
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_norm_size': 64,
     'max_pool_size': (1, 1)},
    {'filters': 128,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_norm_size': 128,
     'max_pool_size': (2, 1)}
]

identifier = 'p1'
player = PPOPlayer(identifier=identifier, max_num_unit_packet=3, observation_history_length=256,
                   sensing_unit_packet_length_ratio=24, unit_packet_success_reward=10,
                   unit_packet_failure_reward=-40, dnn_layers_list=dnn_layers_list, sensing_discount_factor=0.9,
                   dnn_learning_rate=5e-5, scenario=3, modelNumber=0, clipping_val=0.2, entropy_beta=0.01,
                   lmbda=1, value_function_coef=0.5)
player.connect_to_server('127.0.0.1', 8000)
player.train_dnn(num_episodes=100, replay_memory_size=2000, mini_batch_size=1000, dnn_epochs=10,
                 progress_report=True, test_run_length=2000)
