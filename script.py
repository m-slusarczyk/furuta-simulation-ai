#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Self-contained example of using Reinforcement Learning (RL) for
#           a Furuta pendulum. The pendulum is formulated in MINIMAL
#           coordinates.
# Author:   Michał Ślusarczyk based on code by Sebastian Weyrer 
# Date:     2026-03-19
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#!pip install stable-baselines3

import exudyn as exu
from exudyn.utilities import *
from exudyn.artificialIntelligence import *
from stable_baselines3 import DQN, A2C, SAC
import numpy as np
from math import pi
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%% define general behaviour of learning and script
# main settings (GLOBAL variables)
doLearning = True                           # if False, just a saved agent will be loaded, if True, an agent is trained
learningSteps = 10000                         # total number of step updates
stepUpdateTime = 0.02                        # [s] time between the agent selecting actions
integrationStepsPerStepUpdate = 2           # integration steps done between the agent selecting actions

#+++++++++++++++++++++++++++++++++++++++++++++
# settings for learning
tauMax = 15                               # [Nm] max applied torque by the agent
angleThreshold = 0.5                       # [rad] if arm or link angle is above, environment is reset
# upper boundary in which the environmnet is initialized after being reset
randomInitializationUpperBoundary = np.array([0.6, 0, 0.6, 0])

#+++++++++++++++++++++++++++++++++++++++++++++
RLAlgorithm = 'SAC'                         # used RL Algorithm (currently SAC is supported and tested within this script)
modelName = 'solution/agent'                # name of the saved agent
verbose = True                              # set False if no console output should be written during learning
writeDebugFile = False                      # set True for debugging
exportFrames = False                        # set True if you want to export frames of trained agent for nice video

#%% set up logging for TensorboardCallback
from stable_baselines3.common.callbacks import BaseCallback
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose) # the level of verbosity here also affects the logger!
    # class method that is called prior to every rollout (start of the steps used to update the policy)
    def _on_rollout_end(self):
        return True
    # this is called by the RL algorithm DIRECTLY AFTER each call of the step method
    # we must overwrite the @abstractmethod to inherit from BaseCallback
    def _on_step(self):
        # the following is ALREADY done in the non-asbtract class, while we overwrite the abstract class
        # self.n_calls += 1
        # self.num_timesteps = self.model.num_timesteps
        if (self.n_calls % 1e3) == 0 and verbose:
            print('Custom progress info:', str(round((self.n_calls/learningSteps)*100)) + '% of learning finished')
        return True

#%% robotLIDAR
class FurutaEnv(OpenAIGymInterfaceEnv):
    #**classFunction: OVERRIDE this function to create multibody system mbs and setup simulationSettings; call Assemble() at the end!
    #                 you may also change SC.visualizationSettings() individually; kwargs may be used for special setup
    def CreateMBS(self, SC, mbs, simulationSettings, **kwargs):
        # define parameters of the MBS
        gVector = [0, -g, 0]
        mL = 1
        mA = 1
        lL = 1
        lA = 1
        d = 0.2
        inertiaArm = InertiaCuboid(density=mA/(lA*d**2), sideLengths=[d, d, lA])
        inertiaTensorArmAtCOM = inertiaArm.InertiaCOM() # at COM w.r.t. arm frame
        inertiaLink = InertiaCuboid(density=mL/(lL*d**2), sideLengths=[d, lL, d])
        inertiaTensorLinkAtCOM = inertiaLink.InertiaCOM() # at COM w.r.t. link frame
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        # do this once upon setting up the MBS
        self.debugFile = 'debugFile.txt'
        if writeDebugFile:
            with open(self.debugFile, 'w') as file:
                pass
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        # set up graphics for ground and set up graphics for Object Kinematic Tree (OKT)
        oGround = mbs.CreateGround(referencePosition=[0, 0, 0])
        gList = [[[], GraphicsDataOrthoCubePoint(centerPoint=[0, 0, 0], size=[0.5, 0.2, 0.1], color=color4grey)]]
        gJoint = GraphicsDataSphere(point=[0, 0, 0], radius=0.1, color=color4darkgrey)
        gBase = GraphicsDataCylinder(pAxis=[0, -1, 0], vAxis=[0, 1, 0], radius=0.1, color=color4darkgrey)
        gJointAndBase = MergeGraphicsDataTriangleList(gJoint, gBase)
        gList = [[gJointAndBase, GraphicsDataOrthoCubePoint(centerPoint=[0, 0, lA/2], size=[d, d, lA], color=color4grey)]]
        gJoint = GraphicsDataCylinder(pAxis=[0, 0, -0.1], vAxis=[0, 0, 0.2], radius=0.08, color=color4blue)
        # added some offsets for better visualization
        gLink =  GraphicsDataOrthoCubePoint(centerPoint=[0, lL/2, 0], size=[d, lL+d, d*0.99], color=color4red)
        gList += [[gJoint,gLink]]
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        # set up object kinematic tree
        self.nGeneric = mbs.AddNode(NodeGenericODE2(referenceCoordinates=[0]*2,
                                                    initialCoordinates=[0]*2,
                                                    initialCoordinates_t=[0]*2,
                                                    numberOfODE2Coordinates=2))
        self.oKT = mbs.AddObject(ObjectKinematicTree(nodeNumber=self.nGeneric,
                                                     jointTypes=[exu.JointType.RevoluteY]+[exu.JointType.RevoluteZ],
                                                     linkParents=np.arange(-1, 1, 1), # always ascending order
                                                     jointTransformations=exu.Matrix3DList([np.eye(3)]+[np.eye(3)]), # all joints are in same reference system
                                                     # first entry must be [0, 0, 0] as the first joint has no offset
                                                     jointOffsets=exu.Vector3DList([np.array([0, 0, 0])]+[np.array([0, 0, lA])]),
                                                     # list of vectors for inertias at center of mass (COM) in joint/link coordinates; must be always set
                                                     linkInertiasCOM=exu.Matrix3DList([inertiaTensorArmAtCOM]+[inertiaTensorLinkAtCOM]),
                                                     linkCOMs=exu.Vector3DList([np.array([0, 0, lA/2])]+[np.array([0, lL/2, 0])]),
                                                     linkMasses=[mA, mL],
                                                     baseOffset = [0, 0, 0],
                                                     gravity=gVector,
                                                     visualization=VObjectKinematicTree(graphicsDataList=gList)))
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        # assemble multibody system, do basic settings, and return state size
        mbs.Assemble()
        simulationSettings.timeIntegration.numberOfSteps = int(integrationStepsPerStepUpdate) # IMPORTANT: here we set the number of integration steps that should be done in one environment update
        simulationSettings.timeIntegration.endTime = 0 # will be overwritten in step
        simulationSettings.timeIntegration.verboseMode = 0
        simulationSettings.solutionSettings.writeSolutionToFile = False
        simulationSettings.timeIntegration.newton.useModifiedNewton = True
        self.stepUpdateTime = stepUpdateTime # make this accessible for the INTEGRATE STEP method
        return 4
    
    #**classFunction: internal function which initializes dynamic solver; adapt in special cases; this function has some overhead and should not be called during reset() or step()
    def PreInitializeSolver(self):
        self.SetSolver(solverType=exu.DynamicSolverType.RK67) # special solver since we are using minimal coordinates here

    #**classFunction: OVERRIDE this function to set up self.action_space and self.observation_space
    def SetupSpaces(self):
        high = np.array([2*pi, np.finfo(np.float32).max, 2*pi, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-tauMax, high=tauMax, dtype=np.float32)

    #**classFunction: this function is overwritten to map the action given by learning algorithm to the multibody system (environment)
    def MapAction2MBS(self, action):
        tau = action[0]
        self.mbs.SetObjectParameter(self.oKT, 'jointForceVector', [tau, 0])

    #**classFunction: this function is overwrritten to collect output of simulation and map to self.state tuple
    #**output: return bool done which contains information if system state is outside valid range
    def Output2StateAndDone(self):
        q = self.mbs.GetNodeOutput(nodeNumber=self.nGeneric, variableType=exu.OutputVariableType.Coordinates,
                                   configuration=exu.ConfigurationType.Current)
        q_t = self.mbs.GetNodeOutput(nodeNumber=self.nGeneric, variableType=exu.OutputVariableType.Coordinates_t,
                                   configuration=exu.ConfigurationType.Current)
        # array assembling
        self.state = tuple([q[0], q_t[0], q[1], q_t[1]])
        done = False
        # done signal setting, both angles need to be smaller than treshold (so bigger than treshold is changing done to true)
        if abs(q[0]) > angleThreshold or abs(q[1]) > angleThreshold:
            done = True
        if writeDebugFile:
            with open(self.debugFile, 'a') as f:
                f.write('state:\t' + str(self.state) + ' done:\t' + str(done) + '\n')
        return done
    
    # class method that maps the current state to mbs initial values
    def State2InitialValues(self):
        # mapping to get the (redundant) displacement coordinates out of the state of the mbs
        # on velocity base there are no displacement coordinates, but only absolute ones
        # here we are using a minimal coordinates formulation (explicit integration), thus the current coordinates are the system's minimal coordinates
        # the reference coordinates must be substracted, if others than displacements are used in the state (or in the computed redundant coordinates)
        currentDisplacementCoordinates = self.state[0::2]
        currentCoordinates_t = self.state[1::2]
        return [currentDisplacementCoordinates, currentCoordinates_t]
    
    #**classFunction: openAI gym function which resets the system
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # initialize minimal coordinates of mbs (only these are CHANGEABLE initial values of the mechanical system)
        changeableInitialValues = np.random.uniform(low=-randomInitializationUpperBoundary,
                                                    high=randomInitializationUpperBoundary, size=4)
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        # reset other things of the environment
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        # initialize (overwrite) STATE of environment
        self.state = changeableInitialValues # here the state are the changeable initial values

        #+++++++++++++++++++++++++++++++++++++++++++++
        # initialize solver
        self.stepsBeyondDone = None # needed for correct reward zeroing
        self.simulationSettings.timeIntegration.endTime = 0
        self.dynamicSolver.InitializeSolver(self.mbs, self.simulationSettings) # needed to update initial conditions
        # and SET DISPLACEMENT COORDINATES as INITIAL for the FIRST integration step
        [currentDisplacementCoordinates, currentCoordinates_t] = self.State2InitialValues()
        self.mbs.systemData.SetODE2Coordinates(currentDisplacementCoordinates, exu.ConfigurationType.Initial)
        self.mbs.systemData.SetODE2Coordinates_t(currentCoordinates_t, exu.ConfigurationType.Initial)
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        # return state
        return self.state, {}

    #**classFunction: openAI gym interface function which is called to compute one step
    def step(self, action):
        # do main steps
        self.MapAction2MBS(action)
        self.IntegrateStep()
        done = self.Output2StateAndDone()
        reward = 1.0 - (abs(self.state[0]) + abs(self.state[2])) / (2 * angleThreshold)
        # Penalty for falling
        if done:
            reward = -15.0 
        info = {} # do not change this
        return np.array(self.state, dtype=np.float32), reward, done, False, info

    # function to test multibody system and (trained) model
    def TestModel(self, numberOfSteps=500, seed=0, model=None, solutionFileName=None,
                  useRenderer=True, sleepTime=0.01, stopIfDone=False, showTimeSpent=False, makePlots=False, **kwargs):
        import time
        writeToFile = solutionFileName != None
        self.simulationSettings.solutionSettings.writeSolutionToFile = writeToFile
        if writeToFile:
            self.simulationSettings.solutionSettings.coordinatesSolutionFileName = solutionFileName
        storeRenderer = self.useRenderer 
        self.useRenderer = useRenderer #set this true to show visualization
        self.flagNan = False
        observation, info = self.reset(seed=seed, return_info=True)
        ts = -time.time()
        if makePlots:
            observations = observation
        for i in range(numberOfSteps):
            if model != None: #use model to predict action (e.g., controller)
                action, _state = model.predict(observation, deterministic=True)
            else:
                action = np.array([0])
            # print(action)
            if np.isnan(self.state).any(): 
                self.flagNan = True
                break
            observation, reward, done, _, info = self.step(action)
            if makePlots:
                observations = np.vstack((observations, observation))
            if verbose:
                print('--------------------------------')
                print('Testing info at step', i)
                print('r =', reward, '\no =', observation)
            self.render()
            if self.mbs.GetRenderEngineStopFlag(): #user presses quit
                break
            if stopIfDone and done:
                observation, info = self.reset(return_info=True)
            if useRenderer and sleepTime!=0:
                time.sleep(sleepTime)        
        if showTimeSpent:
            print('time spent =', ts + time.time())
        if makePlots:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, sharex=True)
            t = np.linspace(0, numberOfSteps*self.stepUpdateTime, numberOfSteps+1)
            axs[0].plot(t, observations[:, 0])
            axs[0].set_ylabel('arm angle in rad')
            axs[1].plot(t, observations[:, 2])
            axs[1].set_ylabel('link angle in rad')
            axs[1].set_xlabel('time in s')
            for ax in axs:
                ax.grid()
        self.close()
        self.useRenderer = storeRenderer #restore
    
    #**classFunction: openAI gym interface function to render the system
    def render(self, mode="human"):
        if self.rendererRunning == None and self.useRenderer:
            self.SC.renderer.Start()
            self.SC.renderer.DoIdleTasks()
            self.rendererRunning = True
    
    #**classFunction: openAI gym interface function to close system after learning or simulation
    def close(self):
        self.dynamicSolver.FinalizeSolver(self.mbs, self.simulationSettings)
        if self.rendererRunning==True:
            self.SC.renderer.Stop() #safely close rendering window!

#%% initialize and test
if __name__ == '__main__': # this is only executed when file is direct called in Python
    print('EXUDYN says hello with version', exudyn.config.Version())
    print('Starting Reinforcement Learning task ...')
    solutionFile = 'solution/coordinates.txt'
    if doLearning:
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        # create environment and test it
        learningEnv = FurutaEnv()
        if True:
            # no action applied, since no model is specified
            learningEnv.TestModel(numberOfSteps=int(2/learningEnv.stepUpdateTime),
                                  sleepTime=0.05, useRenderer=True, showTimeSpent=False, stopIfDone=True)
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        # get model    
        model = SAC('MlpPolicy',
                    env=learningEnv,
                    device='cpu',
                    train_freq=1,
                    verbose=verbose)
        if verbose:
            print('Detailed agent information:')
            print(model.policy)
        
        #%% do learning
        print('Start learning ...')
        import time
        ts = -time.time()
        model.learn(total_timesteps=int(learningSteps),
                    log_interval=int(1), # yes we want to log and we want to use SB3 logger for it
                    callback=CustomCallback())
        print('Learning time total =', str(round(ts + time.time(), 3)) + 's')
        model.save(modelName)
    
    #%% load and test model
    model = SAC.load(modelName)
    testingEnv = FurutaEnv()
    testingEnv.TestModel(numberOfSteps=int(10/learningEnv.stepUpdateTime), model=model,
                         useRenderer=True, stopIfDone=True, solutionFileName=solutionFile, makePlots=True)
