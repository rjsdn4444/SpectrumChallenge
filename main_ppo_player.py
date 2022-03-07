from player.wifi_ppo_player import PPOPlayer
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
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'max_pool_size': (2, 1)},
    {'filters': 64,
     'kernel_size': (4, 1),
     'strides': (1, 1),
     'max_pool_size': (1, 1)}
]

identifier = 'p1'
player = PPOPlayer(identifier=identifier, max_num_unit_packet=3, observation_history_length=256,
                   sensing_unit_packet_length_ratio=24, unit_packet_success_reward=10,
                   unit_packet_failure_reward=-40, dnn_layers_list=dnn_layers_list, dnn_learning_rate=0.001,
                   clipping_val=0.2, critic_discount=0.9, entropy_beta=0.01, gamma=0.99, lmbda=0.9,
                   scenario=0, model_number=8)
player.connect_to_server('127.0.0.1', 8000)
player.train_dnn(num_episodes=100, dnn_epochs=10, progress_report=True, replay_memory_size=1000, test_run_length=1000)
