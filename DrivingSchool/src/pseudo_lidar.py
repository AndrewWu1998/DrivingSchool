"""
    Description: 将全局二值地图转换为numpy数组形式，用于计算【车辆位置到小目标距离】以及【车辆距离所在道路的边界的值】
"""

import numpy as np
import random
import copy

PSEUDO_LIDAR_RAY_MAX_LENGTH = 10

class PseudoLidarRay:
    def __init__(self, start, theta):
        # theta是与y轴的夹角,是角度制
        # a, b, c是射线所在直线的解析式， ax+by+c = 0
        # x, y 是射线方向的单位向量
        self.start = start
        self.theta = np.deg2rad(theta)
        self.x = np.sin(self.theta)
        self.y = np.cos(self.theta)
        self.a = self.y
        self.b = -self.x
        self.c = -(self.start.x*self.a + self.start.y*self.b)

    def rayStartPoint_to_road_distance(self, road_shape):
        ray_center_to_road_length_list = []

        for equation in road_shape:
            equation_a, equation_b, equation_c = equation[0][0], equation[0][1], equation[0][2]
            equation_m, equation_n = equation[1][0], equation[1][1]
            if equation_a == 0 and equation_b != 0:
                y = equation_c
                x = (-self.b*y - self.c) / self.a
                if equation_m-1e-10 <= x and x <= equation_n+1e-10:
                    ray_center_to_border_length = np.sqrt((self.start.x-x)**2 + (self.start.y-y)**2)
                else:
                    ray_center_to_border_length = float('inf')
            elif equation_a != 0 and equation_b == 0:
                x = equation_c
                y = (-self.a*x - self.c) / (self.b+1e-10) # 防止被 0 除
                if equation_m-1e-10 <= y and y <= equation_n+1e-10:
                    ray_center_to_border_length = np.sqrt((self.start.x-x)**2 + (self.start.y-y)**2)
                else:
                    ray_center_to_border_length = float('inf')
            else:
                pass
            ray_center_to_road_length_list.append(ray_center_to_border_length)

        rayStartPoint_to_road_distance = min(ray_center_to_road_length_list)

        return rayStartPoint_to_road_distance

# def rayStartPoint_to_s_distance(self, road_shape):


        
class MapPseudoLidar:
    """ 由多条 ray 组成，每条 ray 固定间隔，PseudoLidarRay 共同构成 MapPseudoLidar """
    def __init__(self, actor):
        self.actor = actor
    
    def generate_pseudo_lidar_rays(self, left_start_angle=-130, right_end_angle=140, ray_interval_density=10):
        """
            根据需要，规划出 agent 发射出的所有pseudo lidar ray
            
            注意: 设置此处的ray的密度需要在算法 obs size 做对应更改

            Args:
                left_start_angle: 左起射线与中线夹角
                right_end_angle: 最右射线与中线夹角
                ray_interval_density: 射线密度，即每条射线见的夹角
            
            Return: 固定间隔下射线ray的集合
        """
        pseudo_lidar_rays_list = []
        
        for interval_angle in range(left_start_angle, right_end_angle, ray_interval_density):
            pseudo_lidar_rays_list.append(PseudoLidarRay(start=self.actor.loc, theta=(self.actor.forward_azimuth+interval_angle)))

        return pseudo_lidar_rays_list
    
    def pseudo_road_lidar(self, road_shape, maxl=PSEUDO_LIDAR_RAY_MAX_LENGTH):
        """
            用于探测 道路边界 的 lidar ray

            Args:
                maxl (int): lidar最大探测距离

            Return: 射线探测到道路边界距离数值的集合

        """
        pseudo_lidar_rays_list = self.generate_pseudo_lidar_rays(left_start_angle=-130, right_end_angle=140, ray_interval_density=10)

        pseudo_road_lidar_data = []
        for ray in pseudo_lidar_rays_list:
            to_curve_distance = ray.rayStartPoint_to_road_distance(road_shape)
            if to_curve_distance < maxl:
                pseudo_road_lidar_data.append((maxl-to_curve_distance)/maxl)
            else:
                pseudo_road_lidar_data.append(0)

        return pseudo_road_lidar_data



if __name__ == "__main__":

    pass