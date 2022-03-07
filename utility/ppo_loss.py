import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss


class ppo_loss(Loss):
    def __init__(self, clipping_val, critic_discount, entropy_beta, oldpolicy_probs, advantages, rewards, values):
        super(ppo_loss, self).__init__()
        self.clipping_val = clipping_val
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta

        self.oldpolicy_probs = oldpolicy_probs
        self.advantages = advantages
        self.rewards = rewards
        self.values = values

    def call(self, y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(self.oldpolicy_probs + 1e-10))
        general_loss = ratio * self.advantages
        clipped_loss = K.clip(ratio, min_value=1 - self.clipping_val, max_value=1 + self.clipping_val) * self.advantages
        actor_loss = -K.mean(K.minimum(general_loss, clipped_loss))
        critic_loss = K.mean(K.square(self.rewards - self.values))
        total_loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10))
        )
        return total_loss
