import os
# import logging

from gym.envs.registration import register

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "yes please"
# LOG_DIR = os.path.join(os.getcwd(), "logs")
# if not os.path.isdir(LOG_DIR):
#     os.mkdir(LOG_DIR)

# Init and setup the root logger
# logging.basicConfig(filename=LOG_DIR + '/Gymcarla.log', level=logging.DEBUG)

# Author: Estciven
register(
	id="DrivingSchool-v0",
	entry_point='DrivingSchool.envs.DrivingSchool_env:DrivingSchoolEnv'
)