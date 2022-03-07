from player.predict_observation import predictObservation
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

identifier = 'p1'
player = predictObservation(identifier=identifier, observation_history_length=256, dnn_learning_rate=0.001, scenario=1,
                            modelNumber=10)
player.connect_to_server('127.0.0.1', 8000)
player.train_dnn(num_episodes=100, replay_memory_size=10000, mini_batch_size=1000, dnn_epochs=3, progress_report=True,
                 test_run_length=10)
