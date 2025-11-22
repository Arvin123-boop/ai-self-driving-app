import cv2
import numpy as np
import random

class RoadEnv:
    def __init__(self, width=500, height=600):
        self.width = width
        self.height = height
        self.road = np.ones((height, width, 3), dtype=np.uint8) * 200
        self.car_img = cv2.imread("ferrari.png", cv2.IMREAD_UNCHANGED)
        self.car_size = (50, 100)
        self.car_img = cv2.resize(self.car_img, self.car_size)
        self.high_score = 0
        self.reset()

    def reset(self):
        if hasattr(self, 'score') and self.score > self.high_score:
            self.high_score = self.score
        self.car_pos = [self.width//2 - 25, self.height-120]
        self.obstacles = [[random.randint(50, self.width-50), -100, 50, 50] for _ in range(5)]
        self.score = 0
        return self.get_state()

    def step(self, action):
        # Move car
        if action == 0:
            self.car_pos[0] -= 10
        elif action == 2:
            self.car_pos[0] += 10
        self.car_pos[0] = np.clip(self.car_pos[0], 50, self.width-50)

        # Move obstacles
        for obs in self.obstacles:
            obs[1] += 10
            if obs[1] > self.height:
                obs[1] = -50
                obs[0] = random.randint(50, self.width-50)

        # Check collision
        crashed = False
        car_rect = [self.car_pos[0], self.car_pos[1], self.car_size[0], self.car_size[1]]
        for obs in self.obstacles:
            if self.check_collision(car_rect, obs):
                crashed = True
                break

        reward = 1 if not crashed else -100
        self.score += reward if reward>0 else 0
        if self.score > self.high_score:
            self.high_score = self.score

        state = self.get_state()
        if crashed:
            state = self.reset()
        return state, reward, crashed

    def check_collision(self, car, obs):
        x1, y1, w1, h1 = car
        x2, y2, w2, h2 = obs
        return (x1 < x2+w2 and x1+w1 > x2 and y1 < y2+h2 and y1+h1 > y2)

    def get_frame(self):
        frame = self.road.copy()
        for obs in self.obstacles:
            cv2.rectangle(frame, (obs[0], obs[1]), (obs[0]+obs[2], obs[1]+obs[3]), (0,0,255), -1)
        x, y = self.car_pos
        h, w = self.car_img.shape[:2]
        frame[y:y+h, x:x+w] = self.car_img
        return frame

    def get_state(self):
        frame = cv2.resize(self.get_frame(), (84, 84))
        return frame / 255.0
