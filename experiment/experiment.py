import datetime
import math

import carla
import imageio
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

from experiment.base_experiment import BaseExperiment
from helper.carla_helper import post_process_image


class PPOExperiment(BaseExperiment):
    def __init__(self, exp_config):
        super().__init__(exp_config)

        self.action_type = self.exp_config["action_type"]
        self.sensor_type = self.exp_config["sensor_type"]

        self.framestack = self.exp_config["framestack"]
        self.max_time_idle = self.exp_config["max_time_idle"]
        self.max_time_episode = self.exp_config["max_time_episode"]
        self.allowed_types = [carla.LaneType.Driving]
        self.last_heading_deviation = 0
        self.last_action = None

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.collision_impulse = 0
        self.hero_collided = False
        self.done_time_idle = False
        self.done_time_episode = False
        self.done_collision = False
        self.done_reached_the_goal = False

        self.terminated = False
        self.truncated = False

        # Hero variables
        self.last_location = None
        self.last_goal_distance = None
        self.last_velocity = 0

        # Sensor stack
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        self.last_heading_deviation = 0

    def get_action_space(self):
        """Returns the action space, in this case, a discrete space"""
        if self.action_type == "continuous":
            return Box(
                low=np.array([0.0, -0.75, 0.0]), 
                high=np.array([0.75, 0.75, 0.75]), 
                dtype=np.float32
            )
        if self.action_type == "discrete":
            return Discrete(len(self.get_actions()))

    def get_observation_space(self):
        num_channels = 1 if self.exp_config["hero"]["cam_grayscale"] else 3

        if self.sensor_type == "birdview":
            num_cameras = 1
            image_space = Box(
                low=0,
                high=255,
                shape=(
                    self.exp_config["hero"]["sensors"]["birdview"]["top_cam"]["image_size_x"],
                    self.exp_config["hero"]["sensors"]["birdview"]["top_cam"]["image_size_y"],
                    num_cameras * num_channels,
                ),
                dtype=np.uint8,
            )

            return image_space
        
        if self.sensor_type == "fourway_lidar":
            num_cameras = 4
            image_space = Box(
                low=0,
                high=255,
                shape=(
                    self.exp_config["hero"]["sensors"]["fourway_lidar"]["front_cam"]["image_size_x"],
                    self.exp_config["hero"]["sensors"]["fourway_lidar"]["front_cam"]["image_size_y"],
                    num_cameras * num_channels,
                ),
                dtype=np.uint8,
            )

            distance_space = Box(
                low=-1,
                high=self.exp_config["hero"]["sensors"]["fourway_lidar"]["lidar"]["range"],
                shape=(self.exp_config["hero"]["max_detected_actors"],),
                dtype=np.float32,
            )

            obs_space = Dict({"image": image_space, "obj_distance": distance_space})

            return obs_space

    def get_actions(self):
        return {
            0: [0.00, 0.00, 0.00, False, False],  # Coast
            1: [0.00, 0.00, 0.25, False, False],  # Apply Break
            2: [0.00, 0.00, 0.50, False, False],  # Apply Break
            3: [0.00, 0.00, 0.75, False, False],  # Apply Break
            4: [0.00, 0.75, 0.00, False, False],  # Right
            5: [0.00, 0.50, 0.00, False, False],  # Right
            6: [0.00, 0.25, 0.00, False, False],  # Right
            7: [0.00, -0.75, 0.00, False, False],  # Left
            8: [0.00, -0.50, 0.00, False, False],  # Left
            9: [0.00, -0.25, 0.00, False, False],  # Left
            10: [0.25, 0.00, 0.00, False, False],  # Straight
            11: [0.25, 0.75, 0.00, False, False],  # Right
            12: [0.25, 0.50, 0.00, False, False],  # Right
            13: [0.25, 0.25, 0.00, False, False],  # Right
            14: [0.25, -0.75, 0.0, False, False],  # Left
            15: [0.25, -0.50, 0.0, False, False],  # Left
            16: [0.25, -0.25, 0.0, False, False],  # Left
            17: [0.50, 0.00, 0.00, False, False],  # Straight
            18: [0.50, 0.75, 0.00, False, False],  # Right
            19: [0.50, 0.50, 0.00, False, False],  # Right
            20: [0.50, 0.25, 0.00, False, False],  # Right
            21: [0.50, -0.75, 0.0, False, False],  # Left
            22: [0.50, -0.50, 0.0, False, False],  # Left
            23: [0.50, -0.25, 0.0, False, False],  # Left
            24: [0.75, 0.00, 0.00, False, False],  # Straight
            25: [0.75, 0.75, 0.00, False, False],  # Right
            26: [0.75, 0.50, 0.00, False, False],  # Right
            27: [0.75, 0.25, 0.00, False, False],  # Right
            28: [0.75, -0.75, 0.00, False, False],  # Left
            29: [0.75, -0.50, 0.00, False, False],  # Left
            30: [0.75, -0.25, 0.00, False, False],  # Left
        }

    def compute_action(self, action):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero"""
        vehicle_control = carla.VehicleControl()

        if self.action_type == "continuous":
            throttle = (action[0].item() + 1) / 2
            steer = action[1].item()
            brake = (action[2].item() + 1) / 2

            if throttle > brake:
                vehicle_control.throttle = throttle
                vehicle_control.brake = 0
            elif throttle < brake:
                vehicle_control.throttle = 0
                vehicle_control.brake = brake
            vehicle_control.steer = steer
            vehicle_control.reverse = False
            vehicle_control.hand_brake = False

        if self.action_type == "discrete":
            action_control = self.get_actions()[int(action)]

            vehicle_control.throttle = action_control[0]
            vehicle_control.steer = action_control[1]
            vehicle_control.brake = action_control[2]
            vehicle_control.reverse = action_control[3]
            vehicle_control.hand_brake = action_control[4]

        self.last_action = vehicle_control

        return vehicle_control

    def get_observation(self, core, sensor_data):
        """Function to do all the post processing of observations (sensor data).
        Should return a tuple or list of two items, the processed observation ,
        as well as a variable with additional information about such observation."""
        world = core.world
        hero = core.hero
        hero_location = hero.get_location()

        collision = sensor_data.get("collision")
        if collision is not None:
            self.hero_collided = True
            self.collision_impulse = sensor_data["collision"][1][1]
            collision_data = [
                "Object: " + str(sensor_data["collision"][1][0]),
                "Intensity: " + str(sensor_data["collision"][1][1]),
            ]
            with open("collision_history.txt", "a") as f:
                f.writelines("\n".join(collision_data))
                f.write("\n")

        stacked_image = None
        for key in sensor_data.keys():
            if "cam" not in key:
                continue
            id, sensor_reading = sensor_data[key]
            image = post_process_image(
                sensor_reading,
                normalized=self.exp_config["hero"]["cam_normalized"],
                grayscale=self.exp_config["hero"]["cam_grayscale"],
            )
            if stacked_image is None:
                stacked_image = image
            else:
                stacked_image = np.dstack([stacked_image, image])

        if self.sensor_type == "birdview":
            return stacked_image, {}
        
        if self.sensor_type == "fourway_lidar":
            lidar_data = sensor_data["lidar"][1]
            lidar_actor_idx = lidar_data["ObjIdx"]
            actor_idx = []
            for id in lidar_actor_idx:
                if id not in actor_idx and id != hero.id:
                    actor_idx.append(id)

            world_actors = world.get_actors()
            lidar_actors = []
            for id in actor_idx:
                if id != 0:
                    actor = world_actors.find(int(id))
                    lidar_actors.append(actor)

            actor_distance_list = []
            for actor in lidar_actors:
                if actor is not None:
                    actor_location = actor.get_location()
                    actor_distance = float(
                        np.sqrt(
                            np.square(hero_location.x - actor_location.x)
                            + np.square(hero_location.y - actor_location.y)
                        )
                    )
                    actor_distance_list.append(actor_distance)

            max_lidar_actors = self.exp_config["hero"]["max_detected_actors"]
            if len(actor_distance_list) < max_lidar_actors:
                actor_distance_list += [-1] * (max_lidar_actors - len(actor_distance_list))
            else:
                actor_distance_list = sorted(actor_distance_list)[:max_lidar_actors]

            actor_distance_list = np.array(actor_distance_list, dtype=np.float32)

            return {
                "image": stacked_image,
                "obj_distance": actor_distance_list
            }, {}


    def get_hero_velocity(self, hero):
        """Computes the velocity of the hero car in km/h"""
        velocity = hero.get_velocity()
        velocity = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return velocity

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        hero = core.hero

        self.hero_location = hero.get_transform().location
        self.done_reached_goal = core.reached_the_goal(self.hero_location)

        if self.hero_collided:
            self.done_collision = True

        if self.get_hero_velocity(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.done_time_idle = self.max_time_idle < self.time_idle

        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode

        if self.done_collision or self.done_reached_goal:
            self.terminated = True
        if self.done_time_episode or self.done_time_idle:
            self.truncated = True

        return (
            self.terminated,
            self.truncated
        )

    def compute_reward(self, observation, core):
        """Computes the reward"""

        def compute_angle(u, v):
            return -math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])

        world = core.world
        hero = core.hero
        reward = 0

        # Hero-related variables
        hero_location = hero.get_location()
        hero_velocity = self.get_hero_velocity(hero)
        hero_heading = hero.get_transform().get_forward_vector()
        hero_heading = [hero_heading.x, hero_heading.y]
        goal_heading = core.hero_goal_point.get_forward_vector()
        goal_heading = [goal_heading.x, goal_heading.y]

        # Initialize last location
        if self.last_location == None:
            self.last_location = hero_location
        self.goal_location = core.hero_goal_point.location

        goal_distance = float(
            np.sqrt(
                np.square(hero_location.x - self.goal_location.x)
                + np.square(hero_location.y - self.goal_location.y)
            )
        )  # in m
        if self.last_goal_distance == None:
            self.last_goal_distance = goal_distance

        # Compute deltas
        delta_goal_distance = self.last_goal_distance - goal_distance
        delta_velocity = hero_velocity - self.last_velocity  # in m/s

        # Update variables
        self.last_location = hero_location
        self.last_goal_distance = goal_distance
        self.last_velocity = hero_velocity

        closest_waypoint = core.map.get_waypoint(
            hero_location, project_to_road=False, lane_type=carla.LaneType.Any
        )
        if closest_waypoint is None or closest_waypoint.lane_type not in self.allowed_types:
            reward += -1
            self.last_heading_deviation = math.pi
        else:
            if not closest_waypoint.is_junction:
                wp_heading = closest_waypoint.transform.get_forward_vector()
                wp_heading = [wp_heading.x, wp_heading.y]
                angle = compute_angle(hero_heading, wp_heading)
                self.last_heading_deviation = abs(angle)
                if np.dot(hero_heading, wp_heading) < 0:
                    # We are going in the wrong direction
                    reward += -1
                else:
                    if abs(math.sin(angle)) > 0.4:
                        if self.last_action == None:
                            self.last_action = carla.VehicleControl()
                        if self.last_action.steer * math.sin(angle) >= 0:
                            reward += -0.05
            else:
                self.last_heading_deviation = 0

        # Reward if going close to goal
        reward += 10 * delta_goal_distance

        # Reward if going faster than last step
        if hero_velocity > self.last_velocity:
            if hero_velocity < 40:
                reward += 0.5 * delta_velocity
            elif hero_velocity > 50:
                reward += -1 * delta_velocity
        else:
            reward += 0.05 * delta_velocity

        if self.done_time_idle:
            print("Done idle")
            reward += -100
        if self.done_time_episode:
            print("Done max time")
            reward += -100
        if self.done_collision:
            print("Done collided with other object")
            reward += -(100 + 0.5 * self.collision_impulse)
        if self.done_reached_goal:
            print("Done reached the goal")
            reward += 100

        return reward
