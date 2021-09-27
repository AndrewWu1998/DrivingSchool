import gym
import time
import numpy as np

from DrivingSchool.src.ego_vehicle import EgoVehicle
from DrivingSchool.src.pseudo_lidar import MapPseudoLidar

MAX_STEER_RATE = 60

DELTA_TIME = 0.01

PI = 3.14

DEFAULT_CONFIG = {
    "scenarios": {
        "scenariosName": "straight_road" # straight_road corner_road s_road
    },
    "actors": {
        "vehicle_length": 4,
        "vehicle_width": 2
    }
}


class StraightRoad:
    def __init__(self, road_width=6.0, road_length=50.0):
        self.width = road_width
        self.length = road_length
        self.target = [self.width/2.0, self.length-5]
        self.road_shape = [[[0,     1,    0],               [0, road_width]],
                            [[0,    1,    road_length],     [0, road_width]],
                            [[1,    0,    0],               [0, road_length]],
                            [[1,    0,    road_width],      [0, road_length]]]
                           # [a,    b,    c],               [m, n]
                           #  ax  + by  + c  =  0         m < x,y < n
    
    def is_offroad(self, x, y):
        if (0 < x and x < self.width) and (0 < y and y < self.length):
            return False
        else:
            return True
    

class CornerRoad:
    def __init__(self, road_width=6.0, road_length=50.0):
        self.width = road_width
        self.length = road_length
        self.turn = road_length/2.0
        self.road_shape = [[[0,     1,    road_width+self.turn],     [-self.turn, road_width]],
                            [[0,    1,    self.turn],                [-self.turn, 0]],
                            [[0,    1,    0],                        [0, road_width]],
                            [[1,    0,    -self.turn],               [self.turn, road_width+self.turn]],
                            [[1,    0,    0],                        [0, self.turn]],
                            [[1,    0,    road_width],               [0, road_width+self.turn]]]
                           # [a,    b,    c],               [m, n]
                           #  ax  + by  + c  =  0         m < x,y < n

    def is_offroad(self, x, y):
        if (0 < x and x < self.width) and (0 < y and y < self.turn):
            return False
        elif x > 0 and x < (self.length-self.turn) and y > self.turn and y < (self.turn+self.width):
            return False
        else:
            return True


class SnakeRoad:
    def __init__(self, road_width=6.0, road_length=50.0):
        self.width = road_width
        self.length = road_length
        self.road_shape = [[[3,     2/road_length*PI,   0,  0],              [0, road_length]],
                            [[3,    2/road_length*PI,   0,  road_width],     [0, road_length]]]
                        #    [A,           w,           p,  b],             [m, n]
                        # x = A sin(wy + p) + b                             m < x,y < n

        

class DrivingSchoolEnv(gym.Env):
    def __init__(self, configs=None):
        if configs is None:
            configs = DEFAULT_CONFIG

        self.scenarios_name = configs["scenarios"]["scenariosName"]

        self.actor_configs = configs["actors"]

        self.road = None
        self.vehi = None

        self.reset()
        
    def reset(self):
        # 生成道路
        if self.scenarios_name is "straight_road":
            self.road = StraightRoad()
        elif self.scenarios_name is "corner_road":
            self.road = CornerRoad()
        elif self.scenarios_name is "s_road":
            self.road = SnakeRoad()
        else:
            pass

        # 生成ego-vehicle
        self.vehi = EgoVehicle(vehi_loc=[self.road.width/2.0, self.actor_configs["vehicle_length"]/2.0+0.1], actor_config=self.actor_configs, delta_t=DELTA_TIME)

        # 生成pseudo lidar
        self.pseudo_lidar = MapPseudoLidar(actor=self.vehi)
        
        obs, _ = self.read_observation()

        return obs

    def step(self, action):
        throttle = float(np.clip(action[0], -1., 1.))
        steer = float(np.clip(action[1], -1., 1.))
        
        self.vehi.rotate(np.radians(MAX_STEER_RATE*steer)) # 旋转
        self.vehi.forward(throttle) # 平移

        obs, info = self.read_observation()

        reward = None

        done = None

        return obs, reward, done, info
    
    def read_observation(self):
        road_pseudo_lidar_length_list = self.pseudo_lidar.pseudo_road_lidar(self.road.road_shape)

        obs = road_pseudo_lidar_length_list

        measurements = None

        return obs, measurements

    def close(self):
        print("Close")

    def render(self, mode='human'):
        pass


def __main__():
    env = DrivingSchoolEnv()

    try:
        for ep in range(1):
            print("\nEpisode %d:" % (ep))
            obs = env.reset()

            action = [-1, 1]

            start = time.time()
            i = 0
            done = False
            while not done:
                i += 1

                obs, reward, done, info = env.step(action)
                
                print(":{}\n\t".join(["Step#", "obs", "rew", "done", "info:{}"]).format(
                    i, obs, reward, done, info))
                # print("Step# ", i)

                if i > 1:
                    break

            print("{} fps".format(i / (time.time() - start)))
    
    finally:
        env.close()

if __name__ == "__main__":

    __main__()
