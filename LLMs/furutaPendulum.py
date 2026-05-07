#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  This class uses symbolically derived Equations of Motion (EOMs) of
#           the Furuta pendulum and povides some basic functionality
#           with the EOMs and an integrator for the AI Workshop held at LUT.
#           The Furuta pendulum is originally introduced in 
#           https://ieeexplore.ieee.org/document/239008/ by Furuta et al..
# Author:   Michał Ślusarczyk based on code by Sebastian Weyrer 
# Date:     2026-02-23
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import time
import pickle
import sympy as sp
import numpy as np
import exudyn as exu
from exudyn.utilities import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class FurutaPendulum():
    def __init__(self, massArm=1, massLink=1, lengthArm=1, lengthLink=1, width=0.2, gravity=9.81):
        print('Welcome to the "LUT AI Workshop". Setting up the Furuta pendulum...')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        theta_path = os.path.join(script_dir, 'theta_tt.pkl')
        phi_path = os.path.join(script_dir, 'phi_tt.pkl')
        try:
            with open(theta_path, 'rb') as f:
                theta_tts = pickle.load(f)
            with open(phi_path, 'rb') as f:
                phi_tts = pickle.load(f)
        except:
            raise ValueError('Make sure that in the directory, where you run the script, the two files "theta_tt.pkl" and "phi_tt.pkl" are present!')
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # create symbolic expressions such that the class works independent
        thetas, phis = sp.symbols('theta phi')
        theta_ts, phi_ts = sp.symbols('theta_t phi_t')
        mAs, mLs = sp.symbols('m_A m_L')
        lAs, lLs = sp.symbols('l_A l_L')
        ds = sp.symbols('d')
        gs = sp.symbols('g')
        tauAs, tauLs = sp.symbols('tau_A tau_L')
        IACOMys = sp.symbols('I_ACOMy') # scalar inertia for arm around COM in the arm's reference system
        IAP1ys = sp.symbols('I_AP1y') # scalar inertia for arm around pivot 1 in the arm's reference system
        ILCOMxs, ILCOMys, ILCOMzs = sp.symbols('I_LCOMx I_LCOMy I_LCOMz') # spatial inertia for link around COM in the link's reference system
        ILP2xs, ILP2ys, ILP2zs = sp.symbols('I_LP2x I_LP2y I_LP2z')
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # automatically create inertia objects and substitute loaded expressions
        inertiaArm = InertiaCuboid(density=massArm/(lengthArm*width**2), sideLengths=[width, width, lengthArm])
        inertiaTensorArmAtCOM = inertiaArm.InertiaCOM() # at COM w.r.t. arm frame
        inertiaTensorArmAtP1 = inertiaArm.Translated([0, 0, lengthArm/2]).Inertia() # at origin (= P1) w.r.t. arm frame
        inertiaLink = InertiaCuboid(density=massLink/(lengthLink*width**2), sideLengths=[width, lengthLink, width])
        inertiaTensorLinkAtCOM = inertiaLink.InertiaCOM() # at COM w.r.t. link frame
        inertiaTensorLinkAtP2 = inertiaLink.Translated([0, lengthLink/2, 0]).Inertia() # at origin (= P2) w.r.t. link frame
        params = {mAs: massArm,
                  mLs: massLink,
                  lAs: lengthArm,
                  lLs: lengthLink,
                  ds: width,
                  gs: gravity,
                  IACOMys: inertiaTensorArmAtCOM[1, 1],
                  IAP1ys: inertiaTensorArmAtP1[1, 1],
                  ILCOMxs: inertiaTensorLinkAtCOM[0, 0],
                  ILCOMys: inertiaTensorLinkAtCOM[1, 1],
                  ILCOMzs: inertiaTensorLinkAtCOM[2, 2],
                  ILP2xs: inertiaTensorLinkAtP2[0, 0],
                  ILP2ys: inertiaTensorLinkAtP2[1, 1],
                  ILP2zs: inertiaTensorLinkAtP2[2, 2]}
        theta_tt = theta_tts.subs(params)
        phi_tt = phi_tts.subs(params)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # now the EOM are real functions of state and torque, we just have to lambdify them
        self.theta_tt = sp.lambdify((thetas, theta_ts, phis, phi_ts, tauAs, tauLs), theta_tt, 'numpy')
        self.phi_tt = sp.lambdify((thetas, theta_ts, phis, phi_ts, tauAs, tauLs), phi_tt, 'numpy')
        
    def GenerateTrajectory(self, SystemOfODE1=None, x0=[0, 0, 0, 0], simulationTime=10, stepSize=1e-2, plot=False,
                           TauA=None, TauL=None, label='LLM'):
        def _defaultODE(t, x, TauA, TauL):
            dx = np.zeros(4)
            dx[0] = x[1]
            dx[1] = self.theta_tt(x[0], x[1], x[2], x[3], TauA(t, x), TauL(t, x))
            dx[2] = x[3]
            dx[3] = self.phi_tt(x[0], x[1], x[2], x[3], TauA(t, x), TauL(t, x))
            return dx
        if SystemOfODE1 is None:
            print('Using the default EOMs. Not your defined EOMs are used!')
            SystemOfODE1 = _defaultODE # set the system to the default one
            compareWithReference = False
        else:
            print('Your defined EOMs are used.')
            compareWithReference = True
        startTime = time.time()
        t = np.arange(0, simulationTime, stepSize)
        if TauA is None:
            print('No special input torque is used, set arm torque to zero.')
            def TauA(t, x):
                return 0
        if TauL is None:
            print('No special input torque is used, set link torque to zero.')
            def TauL(t, x):
                return 0
        print('Dynamic Simulation via Radau started.')
        sol = solve_ivp(SystemOfODE1,
                        (t[0], t[-1]),
                        x0, t_eval=t,
                        args=(TauA, TauL),
                        method='Radau')
        radauSimulationTime = time.time() - startTime
        if sol['success']:
            print('Equations of Motion solved successfully.')
            print('Time required to solve Equations of Motion: ' + str(radauSimulationTime) + 's.')
        else:
            print('WARNING: The EOMs can not be solved!')
        if compareWithReference:
            startTimeRef = time.time()
            print('Dynamic Simulation via Radau started.')
            solRef = solve_ivp(_defaultODE,
                               (t[0], t[-1]),
                               x0, t_eval=t,
                               args=(TauA, TauL),
                               method='Radau')
            print('Time required to solve reference EOMs: ' + str(time.time() - startTimeRef) + 's.')
        else:
            solRef = None
        if plot:
            figAng, axsAng = plt.subplots(2, 1, sharex=True)
            axsAng[0].plot(t, sol.y[0, :], label=label if compareWithReference else 'radau')
            axsAng[1].plot(t, sol.y[2, :], label=label if compareWithReference else 'radau')
            if solRef is not None:
                axsAng[0].plot(t, solRef.y[0, :], label='reference', linestyle='--')
                axsAng[1].plot(t, solRef.y[2, :], label='reference', linestyle='--')
            axsAng[0].legend()
            axsAng[1].legend()
            axsAng[0].set_ylabel(r'$\theta$ in rad')
            axsAng[1].set_ylabel(r'$\varphi$ in rad')
            axsAng[1].set_xlabel('time in s')
            axsAng[0].grid()
            axsAng[1].grid()
            plt.tight_layout()
            figVel, axsVel = plt.subplots(2, 1, sharex=True)
            axsVel[0].plot(t, sol.y[1, :], label=label if compareWithReference else 'radau')
            axsVel[1].plot(t, sol.y[3, :], label=label if compareWithReference else 'radau')
            if solRef is not None:
                axsVel[0].plot(t, solRef.y[1, :], label='reference', linestyle='--')
                axsVel[1].plot(t, solRef.y[3, :], label='reference', linestyle='--')
            axsVel[0].legend()
            axsVel[1].legend()
            axsVel[0].set_ylabel(r'$\dot\theta$ in rad/s')
            axsVel[1].set_ylabel(r'$\dot\varphi$ in rad/s')
            axsVel[1].set_xlabel('time in s')
            axsVel[0].grid()
            axsVel[1].grid()
            plt.tight_layout()
        if compareWithReference:
            fileName = 'solution'
            header = 't,theta_student,phi_student,theta_reference,phi_reference'
            data = np.column_stack((t, sol.y[0, :], sol.y[2, :], solRef.y[0, :], solRef.y[2, :]))
            np.savetxt(fileName + '.csv', data, delimiter=',', header=header, comments='')
            print('Comparison saved to ' + fileName + '.csv')
        return sol.y

if __name__ == '__main__':
    pendulum = FurutaPendulum()
    def TauA(t, x):
        return 1
    traj = pendulum.GenerateTrajectory(x0=[0, 0, 0.01, 0], plot=True,
                                       TauA=TauA)
