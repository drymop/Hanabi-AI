import numpy as np
from typing import List


class Experience:
    def __init__(self, state, next_state, reward):
        self.state = state
        self.next_state = next_state
        self.reward = reward


class ExperienceBuffer:
    def __init__(self, max_len: int, buffer_prob_scale: float):
        self.max_len = max_len
        self._buffer = [None] * self.max_len  # type: List[Experience]
        self.len = 0
        self._insert_index = 0  # type: int
        self._unnormalized_prob = np.zeros(
            self.max_len)  # proportional to the probability that an experience in sampled
        self._buffer_prob_scale = buffer_prob_scale

    def __len__(self):
        return self.len

    def add(self, experience: Experience):
        self._buffer[self._insert_index] = experience
        self._unnormalized_prob[self._insert_index] = self._calculate_unnormalized_probability(experience.reward)
        self.len = min(self.max_len, self.len + 1)
        self._insert_index = (self._insert_index + 1) % self.max_len

    def _calculate_unnormalized_probability(self, reward):
        return abs(reward) * self._buffer_prob_scale + 1

    def sample(self, size = 1):
        normalized_prob = self._unnormalized_prob / sum(self._unnormalized_prob)
        return np.random.choice(self._buffer, size=size, p=normalized_prob)
