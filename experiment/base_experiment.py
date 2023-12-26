class BaseExperiment:
    def __init__(self, exp_config):
        self.exp_config = exp_config

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        pass

    def get_action_space(self):
        """Returns the action space"""
        raise NotImplementedError
    
    def get_observation_space(self):
        """Returns the observation space"""
        raise NotImplementedError
    
    def get_actions(self):
        """Returns the actions"""
        raise NotImplementedError
    
    def compute_action(self, action):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero"""
        raise NotImplementedError
    
    def get_observation(self, sensor_data):
        """Function to do all the post processing of the observations (sensor data).
        Should return a tuple or list with two items, the processed observations, 
        as well as a variable with additional information about such observation."""
        return NotImplementedError
    
    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        return NotImplementedError
    
    def compute_reward(self, observation, core):
        """Computes the reward"""
        return NotImplementedError