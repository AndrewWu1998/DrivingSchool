import numpy as np

class EgoVehicle:
    def __init__(self, vehi_loc, actor_config, delta_t):
        self.length = actor_config["vehicle_length"]
        self.width = actor_config["vehicle_width"]
        self.loc = Location(vehi_loc[0], vehi_loc[1])
        self.velocity = 0
        self.forward_azimuth = 0 # ego的朝向，单位: 度

        self.bbx = np.zeros((4, 2))
        self.bbx[0, :] = np.array([self.loc.x+self.width/2.0, self.loc.y+self.length/2.0]) # 四个顶角
        self.bbx[1, :] = np.array([self.loc.x+self.width/2.0, self.loc.y-self.length/2.0])
        self.bbx[2, :] = np.array([self.loc.x-self.width/2.0, self.loc.y-self.length/2.0])
        self.bbx[3, :] = np.array([self.loc.x-self.width/2.0, self.loc.y+self.length/2.0])

        self.delta_t = delta_t
    
    def forward(self, throttle):
        move_dis = (self.velocity + throttle) * self.delta_t
        self.loc.x = self.loc.x + move_dis * np.sin(self.forward_azimuth)
        self.loc.y = self.loc.y + move_dis * np.cos(self.forward_azimuth)
        self.bbx = self.bbx + [move_dis * np.sin(self.forward_azimuth), move_dis * np.cos(self.forward_azimuth)]

    def rotate(self, theta):
        self.forward_azimuth += theta
        euler_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        extend = self.bbx - [self.loc.x, self.loc.y]
        extend = np.dot(euler_matrix, extend.T)
        self.bbx = extend.T + [self.loc.x, self.loc.y]

class Location:
    def __init__(self, x, y):
        self.x = x
        self.y = y