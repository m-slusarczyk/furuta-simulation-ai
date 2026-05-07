#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  This script is part of the AI Workshop held at LUT (2026).
# Task:     Large Language Models (LLMs)
# Author:   Michał Ślusarczyk
# Date:     12.03.2026
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import os
import numpy as np
# add parent directory to find special class
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from furutaPendulum import FurutaPendulum

studentNumber = None

massArm = 1         # [kg] mass of the arm
massLink = 1        # [kg] mass of the link
lengthArm = 1       # [m] length of the arm
lengthLink = 1      # [m] lengthof the link
width = 0.2         # [m] width of arm / link
gravity = 9.81      # [m/s^2] gravitational acceleration

def ODE1System(t, x, tauA, tauL): # tauA and tauL (both functions) should NOT appear in your ODEs!
    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = -(( (massLink*lengthLink**2) * (- (-massLink*lengthArm*lengthLink*np.sin(x[2])*x[3]**2 + 2*massLink*lengthLink**2*np.sin(x[2])*np.cos(x[2])*x[1]*x[3])) + (massLink*lengthArm*lengthLink*np.cos(x[2])) * (-massLink*lengthLink**2*np.sin(x[2])*np.cos(x[2])*x[1]**2 - massLink*gravity*lengthLink*np.sin(x[2])) ) / ( (massArm*lengthArm**2 + massLink*lengthArm**2 + massLink*lengthLink**2*np.sin(x[2])**2) * (massLink*lengthLink**2) - (massLink*lengthArm*lengthLink*np.cos(x[2]))**2 ))
    dx[2] = x[3]
    dx[3] = ( - (massLink*lengthArm*lengthLink*np.cos(x[2])) * (- (-massLink*lengthArm*lengthLink*np.sin(x[2])*x[3]**2 + 2*massLink*lengthLink**2*np.sin(x[2])*np.cos(x[2])*x[1]*x[3])) - (massArm*lengthArm**2 + massLink*lengthArm**2 + massLink*lengthLink**2*np.sin(x[2])**2) * (-massLink*lengthLink**2*np.sin(x[2])*np.cos(x[2])*x[1]**2 - massLink*gravity*lengthLink*np.sin(x[2])) ) / ( (massArm*lengthArm**2 + massLink*lengthArm**2 + massLink*lengthLink**2*np.sin(x[2])**2) * (massLink*lengthLink**2) - (massLink*lengthArm*lengthLink*np.cos(x[2]))**2 )
    return dx

pendulum = FurutaPendulum(massArm=massArm, massLink=massLink,
                          lengthArm=lengthArm, lengthLink=lengthLink,
                          width=width, gravity=gravity)
x0 = [0, 0, 0.01, 0]
pendulum.GenerateTrajectory(SystemOfODE1=ODE1System, x0=x0,
                            label='EOMs by LLM', plot=True, TauA=None)

