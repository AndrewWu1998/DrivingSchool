import gym
import time
import numpy as np
import pygame

from DrivingSchool.src.ego_vehicle import EgoVehicle
from DrivingSchool.src.pseudo_lidar import MapPseudoLidar

MAX_STEER_RATE = 1

DELTA_TIME = 0.01

PI = 3.14

RENDER_SCALE = 20

COLOR_BUTTER_2 = pygame.Color(196, 160, 0)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)

DEFAULT_CONFIG = {
    "scenarios": {
        "scenariosName": "straight_road" # straight_road corner_road s_road
    },
    "env": {
        "render": False
    }
    "actors": {
        "vehicle_length": 4,
        "vehicle_width": 2
    }
}


class StraightRoad:
    def __init__(self, road_width=3.0, road_length=50.0):
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
        self.env_configs = configs["env"]
        self.actor_configs = configs["actors"]

        self.observation_space = np.zeros(27)
        self.action_space = np.zeros(2)

        self.road = None
        self.vehi = None
        self.pseudo_lidar = None
        self.start_loc = None
        self.curr_measurement = None
        self.prev_measurement = None

        self.display = None
        self.pygame_clock = None

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
        
        # pygame init
        if self.env_configs["render"]:
            pygame.init()
            self.display = pygame.display.set_mode((int(self.road.width*RENDER_SCALE)*3, int(self.road.length*RENDER_SCALE)), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0,0,0))
            pygame.draw.rect(self.display, COLOR_ALUMINIUM_3, (self.road.width*RENDER_SCALE, 0, self.road.width*RENDER_SCALE, self.road.length*RENDER_SCALE), 0)
            pygame.display.flip()
            self.pygame_clock = pygame.time.Clock()
        
        # 生成ego-vehicle
        self.start_loc = [self.road.width/2.0, self.actor_configs["vehicle_length"]/2.0+0.1]
        self.vehi = EgoVehicle(vehi_loc=self.start_loc, actor_config=self.actor_configs, delta_t=DELTA_TIME)

        # 生成pseudo lidar
        self.pseudo_lidar = MapPseudoLidar(actor=self.vehi)
        
        obs, measurement = self.read_observation()
        self.prev_measurement = measurement

        return obs

    def step(self, action):
        throttle = float(np.clip(action[0], -1., 1.))
        steer = float(np.clip(action[1], -1., 1.))
        
        self.vehi.rotate(np.radians(MAX_STEER_RATE*steer)) # 旋转
        move_distance = self.vehi.forward(throttle) # 平移

        obs, self.curr_measurement = self.read_observation()

        reward = 0

        offroad = False
        for point in self.vehi.bbx:
            offroad = offroad or self.road.is_offroad(point[0], point[1])

        low_speed_alongRoad = False
        if self.vehi.velocity*np.cos(self.vehi.forward_azimuth) < 0.1:
            low_speed_alongRoad = True

        wrong_direction = False
        if abs(self.vehi.forward_azimuth) > 90:
            wrong_direction = True

        done = offroad or low_speed_alongRoad or wrong_direction

        self.curr_measurement["done"] = {
            "offroad": offroad,
            "low_speed_alongRoad": low_speed_alongRoad,
            "wrong_direction": wrong_direction
        }

        return obs, reward, done, self.curr_measurement
    
    def read_observation(self):
        road_pseudo_lidar_length_list = self.pseudo_lidar.pseudo_road_lidar(self.road.road_shape)

        obs = road_pseudo_lidar_length_list

        measurements = {
            "location": [float(format(self.vehi.loc.x, '.4f')), float(format(self.vehi.loc.y, '.4f'))],
            "total_distance": np.sqrt((self.vehi.loc.x - self.start_loc[0])**2 + (self.vehi.loc.y - self.start_loc[1])**2),
            "velocity": self.vehi.velocity
        }
        
        return obs, measurements

    def close(self):
        print("Close")

        if self.env_configs["render"]:
            pygame.quit()

    def render(self):
        if self.env_configs["render"]:
            self.pygame_clock.tick_busy_loop(60)
            bbx_list = [(self.road.width*RENDER_SCALE+self.vehi.bbx[0][0]*RENDER_SCALE, self.vehi.bbx[0][1]*RENDER_SCALE), 
                        (self.road.width*RENDER_SCALE+self.vehi.bbx[1][0]*RENDER_SCALE, self.vehi.bbx[1][1]*RENDER_SCALE), 
                        (self.road.width*RENDER_SCALE+self.vehi.bbx[2][0]*RENDER_SCALE, self.vehi.bbx[2][1]*RENDER_SCALE), 
                        (self.road.width*RENDER_SCALE+self.vehi.bbx[3][0]*RENDER_SCALE, self.vehi.bbx[3][1]*RENDER_SCALE)]
            # pygame.draw.rect(self.display, COLOR_BUTTER_2, (self.road.width*RENDER_SCALE+self.vehi.bbx[3][0]*RENDER_SCALE, self.vehi.bbx[3][1]*RENDER_SCALE, self.vehi.width*RENDER_SCALE, self.vehi.length*RENDER_SCALE), 0)
            pygame.draw.polygon(self.display, COLOR_BUTTER_2, bbx_list, 0)
            pygame.display.flip()


def __main__():
    env = DrivingSchoolEnv()

    try:
        for ep in range(1):
            print("\nEpisode %d:" % (ep))
            obs = env.reset()

            action = [0, 0]

            start = time.time()
            i = 0
            done = False
            while not done:
                i += 1

                obs, reward, done, info = env.step(action)
                
                # print(":{}\n\t".join(["Step#", "obs", "rew", "done", "info:{}"]).format(
                #     i, obs, reward, done, info))
                print("Step# ", i)

                env.render()

                if i > 0:
                    break

            # print("{} fps".format(i / (time.time() - start)))
    
    finally:
        env.close()

if __name__ == "__main__":

    __main__()
