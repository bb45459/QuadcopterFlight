import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.03*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        #Distance to goal state
        #reward = 1. - 0.003*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        
        #reward = np.tanh(1 - 0.3*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        #reward = 1. - ((sum(abs(self.sim.pose[:3] - self.target_pos)))**(0.4))
        #reward = -0.003*(abs(self.sim.pose[:3] - self.target_pos)).sum()
       #reward = -min(0.003*(abs(self.target_pos - self.sim.pose[:3])).sum(), 20.0) # reward = zero for matching target z, -ve as you go farther, upto -20
        #if self.sim.pose[2] > self.target_pos[2]-0.2:
        #    reward+=1 #bonus for nearing desired height
        #    print("Close on height!")
        #reward = -0.01*(sum(abs(self.sim.pose[:3] - self.target_pos)))
        
        #if (abs(self.sim.pose[:3] - self.target_pos)).sum() < 6:
        #    print("Close enough!")
        #    reward+=0.1
        
        reward = 0
        penalty = 0
        current_position = self.sim.pose[:3]
        # penalty for euler angles, we want the takeoff to be stable
        penalty += abs(self.sim.pose[3:6]).sum()
        penalty += abs(current_position[0]-self.target_pos[0])**2
        penalty += abs(current_position[1]-self.target_pos[1])**2
        penalty += abs(current_position[2]-self.target_pos[2])**2

        #distance = np.sqrt((current_position[0]-self.target_pos[0])**2 + (current_position[1]-self.target_pos[1])**2 + (current_position[2]-self.target_pos[2])**2)
        #if distance < 10:
        #    reward += 1000

        reward += 100
        return reward - penalty*0.002
        
        #return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        
        next_state = np.concatenate(pose_all)
        if (abs(self.sim.pose[:3] - self.target_pos)).sum() < 5:
            print("Made it to takeoff zone!")
            reward+=1000
            done=True
            
        if self.sim.pose[2]<80:
            print("Went down")
            reward-=10000
            done=True
            
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state