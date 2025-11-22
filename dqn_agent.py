import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random
from collections import deque

class DQNAgent:
    def __init__(self, action_size=3):
        self.state_size = (84, 84, 3)
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=self.state_size),
            layers.Conv2D(64, (4,4), strides=(2,2), activation='relu'),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
            target_f = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
