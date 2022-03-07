from player.wifi_ppo_player_minibatch import PPOPlayer
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


dnn_layers_list = [
    {'filters': 32,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_normalize': None,
     'max_pool_size': (2, 1)},
    {'filters': 32,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_normalize': True,
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_normalize': None,
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_normalize': True,
     'max_pool_size': (1, 1)},
    {'filters': 128,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'batch_normalize': None,
     'max_pool_size': (2, 1)}
]

identifier = 'p1'
player = PPOPlayer(identifier=identifier, max_num_unit_packet=3, observation_history_length=256,
                   sensing_unit_packet_length_ratio=24, unit_packet_success_reward=10,
                   unit_packet_failure_reward=-40, dnn_layers_list=dnn_layers_list, sensing_discount_factor=0,
                   dnn_learning_rate=0.001, scenario=1, modelNumber=0, clipping_val=0.3, entropy_beta=0.01,
                   lmbda=1, value_function_coef=0.5)
player.connect_to_server('127.0.0.1', 8000)
player.train_dnn(num_episodes=100, replay_memory_size=2000, mini_batch_size=1000, dnn_epochs=10,
                 progress_report=True, test_run_length=2000)
