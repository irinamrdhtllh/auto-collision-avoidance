import logging
import os
import random
import subprocess
import time

import carla
import numpy as np

from helper.carla_helper import is_used
from helper.sensors.sensor_factory import SensorFactory
from helper.sensors.sensor_interface import SensorInterface


class CarlaCore:
    def __init__(self, carla_config, exp_config):
        self.carla_config = carla_config
        self.exp_config = exp_config

        self.client = None
        self.world = None
        self.map = None
        self.traffic_manager = None

        self.hero = None
        self.hero_spawn_point = None
        self.opposite_spawn_point = None
        self.hero_goal_point = None
        self.has_reached_the_goal = False

        self.sensor_interface = SensorInterface()

        self.mode = self.exp_config["mode"]
        self.scenario = self.exp_config["scenario"]
        self.sensor_type = self.exp_config["sensor_type"]

        self.actors = []

        self.cars_list = []
        self.cars_id = []

        self.walkers_list = []
        self.all_id = []
        self.all_actors = []

        self.danger_car = None
        self.danger_cars_list = []
        self.danger_car_spawn_point = None
        self.danger_car_route = []

        self.init_server()
        self.connect_client()

    def init_server(self):
        self.server_port = random.randint(15000, 32000)

        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + str(self.server_port))
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port + 1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port + 1)

        if self.carla_config["show_display"]:
            server_command = [
                "{}\CarlaUE4.exe".format(os.environ["CARLA_ROOT"]),
                "-windowed",
                "-ResX={}".format(self.carla_config["res_x"]),
                "-ResY={}".format(self.carla_config["res_y"]),
                "-quality-level={}".format(self.carla_config["quality_level"]),
            ]
        else:
            server_command = [
                "{}\CarlaUE4.exe".format(os.environ["CARLA_ROOT"]),
                "-RenderOffScreen",
            ]

        server_command += [
            "--carla-rpc-port={}".format(self.server_port),
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        """Connect to the client"""
        for i in range(self.carla_config["retries_on_error"]):
            try:
                self.client = carla.Client(self.carla_config["host"], self.server_port)
                self.client.set_timeout(self.carla_config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.carla_config["enable_rendering"]
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.carla_config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return
            except Exception as e:
                print(
                    " Waiting for server to be ready: {}, attempt {} of {}".format(
                        e, i + 1, self.carla_config["retries_on_error"]
                    )
                )
                time.sleep(5)

        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration"
        )

    def setup_experiment(self):
        self.world = self.client.load_world(
            map_name=self.exp_config["town"][self.mode],
            reset_settings=False,
            map_layers=carla.MapLayer.All
            if self.carla_config["enable_map_assets"]
            else carla.MapLayer.NONE,
        )

        self.map = self.world.get_map()

        weather = getattr(carla.WeatherParameters, self.exp_config["weather"])

        self.world.set_weather(weather)

        self.tm_port = 8000
        while is_used(self.tm_port):
            print(
                "Traffic manager's port "
                + str(self.tm_port)
                + " is already being used. Checking the next one"
            )
            self.tm_port += 1
        print("Traffic manager connected to port " + str(self.tm_port))

        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_hybrid_physics_mode(
            self.exp_config["traffic"]["tm_hybrid_mode"]
        )

        seed = self.exp_config["traffic"]["seed"]
        if seed is not None:
            self.traffic_manager.set_random_device_seed(seed)

    def reset_hero(self):
        """This function resets/spawns the hero vehicle and its sensors"""
        # Destroy hero
        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        # Destroy all sensors
        self.sensor_interface.destroy()

        self.world.tick()

        # Spawn the hero car
        spawn_points = self.map.get_spawn_points()
        lane_list = self.exp_config["hero"]["spawn_points"][self.mode]
        hero_lane = random.choice(lane_list)
        spawn_point_index = random.choice(hero_lane)
        self.hero_spawn_point = spawn_point_index
        spawn_point = spawn_points[self.hero_spawn_point]

        if hero_lane == lane_list[0]:
            opposite_lane = lane_list[2]
        elif hero_lane == lane_list[1]:
            opposite_lane = lane_list[3]
        elif hero_lane == lane_list[2]:
            opposite_lane = lane_list[0]
        elif hero_lane == lane_list[3]:
            opposite_lane = lane_list[1]

        self.opposite_spawn_point = random.choice(opposite_lane)

        hero_model = "".join(self.exp_config["hero"]["model"])
        hero_blueprint = self.world.get_blueprint_library().find(hero_model)
        hero_blueprint.set_attribute("role_name", "hero")

        self.hero = self.world.spawn_actor(hero_blueprint, spawn_point)
        if self.hero is None:
            raise AssertionError(
                f"Error spawning hero: {hero_blueprint} at point {spawn_point} index {self.hero_spawn_point}"
            )

        self.world.tick()

        hero_sensors = self.exp_config["hero"]["sensors"][self.sensor_type]

        if self.hero is not None:
            print("Hero spawned!")
            for name, attributes in hero_sensors.items():
                sensor = SensorFactory.spawn(name, attributes, self.sensor_interface, self.hero)

    def reached_the_goal(self, hero_location):
        has_reached_the_goal = False

        goal_points = self.map.get_spawn_points()
        goal_points_per_lane = self.exp_config["hero"]["goal_points"][self.mode]
        goal_bounds = self.exp_config["hero"]["goal_bounds"][self.mode]
        spawn_points_per_lane = self.exp_config["hero"]["spawn_points"][self.mode]

        if self.scenario == "straight":  # go straight
            if self.hero_spawn_point in spawn_points_per_lane[0]:
                goal_point_id = random.choice(goal_points_per_lane[2])
                goal_bound = goal_bounds[2]
            elif self.hero_spawn_point in spawn_points_per_lane[1]:
                goal_point_id = random.choice(goal_points_per_lane[3])
                goal_bound = goal_bounds[3]
            elif self.hero_spawn_point in spawn_points_per_lane[2]:
                goal_point_id = random.choice(goal_points_per_lane[0])
                goal_bound = goal_bounds[0]
            elif self.hero_spawn_point in spawn_points_per_lane[3]:
                goal_point_id = random.choice(goal_points_per_lane[1])
                goal_bound = goal_bounds[1]

        elif self.scenario == "turn_right":  # turn right
            if self.hero_spawn_point in spawn_points_per_lane[0]:
                goal_point_id = random.choice(goal_points_per_lane[1])
                goal_bound = goal_bounds[1]
            elif self.hero_spawn_point in spawn_points_per_lane[1]:
                goal_point_id = random.choice(goal_points_per_lane[2])
                goal_bound = goal_bounds[2]
            elif self.hero_spawn_point in spawn_points_per_lane[2]:
                goal_point_id = random.choice(goal_points_per_lane[3])
                goal_bound = goal_bounds[3]
            elif self.hero_spawn_point in spawn_points_per_lane[3]:
                goal_point_id = random.choice(goal_points_per_lane[0])
                goal_bound = goal_bounds[0]

        elif self.scenario == "turn_left":  # turn left
            if self.hero_spawn_point in spawn_points_per_lane[0]:
                goal_point_id = random.choice(goal_points_per_lane[3])
                goal_bound = goal_bounds[3]
            elif self.hero_spawn_point in spawn_points_per_lane[1]:
                goal_point_id = random.choice(goal_points_per_lane[0])
                goal_bound = goal_bounds[0]
            elif self.hero_spawn_point in spawn_points_per_lane[2]:
                goal_point_id = random.choice(goal_points_per_lane[1])
                goal_bound = goal_bounds[1]
            elif self.hero_spawn_point in spawn_points_per_lane[3]:
                goal_point_id = random.choice(goal_points_per_lane[2])
                goal_bound = goal_bounds[2]

        self.hero_goal_point = goal_points[goal_point_id]

        if goal_bound == goal_bounds[0]:
            if (hero_location.x <= goal_bound[0][1] and 
                hero_location.y >= goal_bound[1][0] and 
                hero_location.y <= goal_bound[1][1]):
                    has_reached_the_goal = True
        elif goal_bound == goal_bounds[1]:
            if (hero_location.x >= goal_bound[0][0] and
                hero_location.x <= goal_bound[0][1] and 
                hero_location.y >= goal_bound[1][0]):
                    has_reached_the_goal = True
        elif goal_bound == goal_bounds[2]:
            if (hero_location.x >= goal_bound[0][0] and 
                hero_location.y >= goal_bound[1][0] and 
                hero_location.y <= goal_bound[1][1]):
                    has_reached_the_goal = True
        elif goal_bound == goal_bounds[3]:
            if (hero_location.x >= goal_bound[0][0] and
                hero_location.x <= goal_bound[0][1] and 
                hero_location.y <= goal_bound[1][1]):
                    has_reached_the_goal = True

        return has_reached_the_goal

    def tick(self, control):
        """Performs one tick of the simulation, moving all actors, and getting the sensor data"""
        # Move hero car
        if control is None:
            pass
        else:
            self.apply_hero_control(control)

        # Tick once the simulation
        self.world.tick()

        # Move the spectator
        if self.carla_config["enable_rendering"]:
            self.set_spectator_camera_view()

        # Return the new sensor data
        return self.get_sensor_data()

    def set_spectator_camera_view(self):
        transform = self.hero.get_transform()

        # Get the camera position
        server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
        server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
        server_view_z = transform.location.z + 3

        # Get the camera orientation
        server_view_roll = transform.rotation.roll
        server_view_yaw = transform.rotation.yaw
        server_view_pitch = transform.rotation.pitch

        # Get the spectator and place it on the desired position
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch, yaw=server_view_yaw, roll=server_view_roll),
            )
        )

    def apply_hero_control(self, control):
        """Applies the control calculated at the experiment to the hero"""
        self.hero.apply_control(control)

    def get_sensor_data(self):
        """Returns the data sent by the different sensors at this tick"""
        sensor_data = self.sensor_interface.get_data()
        return sensor_data

    def spawn_cars(self):
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Get the car blueprints and the number of spawning cars
        vehicle_blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        car_blueprints = [
            x for x in vehicle_blueprints if int(x.get_attribute("number_of_wheels")) == 4
        ]
        number_of_car_actors = random.choice(
            self.exp_config["traffic"]["n_other_cars"]
        )

        # Get the spawn points
        spawn_points = self.map.get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_car_actors < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_car_actors > number_of_spawn_points:
            msg = "Requested %d vehicles, but could only find %d spawn points"
            logging.warning(msg, number_of_car_actors, number_of_spawn_points)
            number_of_car_actors = number_of_spawn_points

        # Before we spawn the actors, make sure we are not spawning in the hero location
        # and in front of the danger car
        spawn_points.remove(spawn_points[self.hero_spawn_point])
        for spawn_point in self.danger_car_route:
            spawn_points.remove(spawn_points[spawn_point])

        # Spawn all the car actors
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_car_actors:
                break

            car_blueprint = random.choice(car_blueprints)
            if car_blueprint.has_attribute("color"):
                color = random.choice(car_blueprint.get_attribute("color").recommended_values)
                car_blueprint.set_attribute("color", color)
            if car_blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    car_blueprint.get_attribute("driver_id").recommended_values
                )
                car_blueprint.set_attribute("driver_id", driver_id)

            batch.append(
                SpawnActor(car_blueprint, transform).then(
                    SetAutopilot(FutureActor, True, self.traffic_manager.get_port())
                )
            )

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                self.cars_list.append(response.actor_id)

    def spawn_walkers(self):
        walker_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")

        number_of_walkers = random.choice(
            self.exp_config["traffic"]["n_walkers"]
        )

        percentage_pedestrian_running = 0.0
        percentage_pedestrian_crossing = 0.0

        SpawnActor = carla.command.SpawnActor

        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            blueprint = random.choice(walker_blueprints)
            if blueprint.has_attribute("is_invicible"):
                blueprint.set_attribute("is_invicible", "false")
            if blueprint.has_attribute("speed"):
                if random.random() > percentage_pedestrian_running:
                    walker_speed.append(blueprint.get_attribute("speed").recommended_values[1])
                else:
                    walker_speed.append(blueprint.get_attribute("speed").recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(blueprint, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed_dummy = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed_dummy.append(walker_speed[i])
        walker_speed = walker_speed_dummy
        
        batch = []
        walker_controller_blueprint = self.world.get_blueprint_library().find(
            "controller.ai.walker"
        )
        for i in range(len(self.walkers_list)):
            batch.append(
                SpawnActor(
                    walker_controller_blueprint, carla.Transform(), self.walkers_list[i]["id"]
                )
            )
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id

        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        self.world.set_pedestrians_cross_factor(percentage_pedestrian_crossing)
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].start()
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            self.all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

    def spawn_danger_car(self):
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        spawn_points = self.world.get_map().get_spawn_points()

        hero_spawn_points_per_lane = self.exp_config["hero"]["spawn_points"][self.mode]

        route_indices_list = []
        if self.scenario == "straight":
            if self.hero_spawn_point in hero_spawn_points_per_lane[0]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][4],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][5],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][6],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][7],
                ]
            elif self.hero_spawn_point in hero_spawn_points_per_lane[1]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][0],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][1],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][2],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][3],
                ]
            elif self.hero_spawn_point in hero_spawn_points_per_lane[2]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][4],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][5],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][6],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][7],
                ]
            elif self.hero_spawn_point in hero_spawn_points_per_lane[3]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][0],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][1],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][2],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][3],
                ]
        elif self.scenario == "turn_right":
            if self.hero_spawn_point in hero_spawn_points_per_lane[0]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][6],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][7],
                ]
            elif self.hero_spawn_point in hero_spawn_points_per_lane[1]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][0],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][1],
                ]
            elif self.hero_spawn_point in hero_spawn_points_per_lane[2]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][4],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][5],
                ]
            elif self.hero_spawn_point in hero_spawn_points_per_lane[3]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][2],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][3],
                ]
        elif self.scenario == "turn_left":
            if self.hero_spawn_point in hero_spawn_points_per_lane[0]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][2],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][3],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][4],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][5],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][6],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][7],
                ]
            elif self.hero_spawn_point in hero_spawn_points_per_lane[1]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][0],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][1],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][2],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][3],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][6],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][7],
                ]
            elif self.hero_spawn_point in hero_spawn_points_per_lane[2]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][0],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][1],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][4],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][5],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][6],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][7],
                ]
            elif self.hero_spawn_point in hero_spawn_points_per_lane[3]:
                route_indices_list = [
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][0],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][1],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][2],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][3],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][4],
                    self.exp_config["traffic"]["danger_car_routes"][self.mode][5],
                ]

        number_of_danger_car = self.exp_config["traffic"]["n_danger_cars"]
        number_of_route = len(route_indices_list)

        vehicle_blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        car_blueprints = [
            x for x in vehicle_blueprints if x.get_attribute("base_type") == "car"
        ]

        batch = []
        route_batch = []
        for n in range(number_of_danger_car):
            if n >= number_of_route:
                break

            danger_car_route = random.choice(route_indices_list)
            route_indices_list.remove(danger_car_route)

            spawn_point = spawn_points[danger_car_route[0]]
            route = []
            for ind in danger_car_route:
                self.danger_car_route.append(ind)
                route.append(spawn_points[ind].location)

            route_batch.append(route)

            danger_car_blueprint = random.choice(car_blueprints)
            if danger_car_blueprint.has_attribute("color"):
                color = random.choice(danger_car_blueprint.get_attribute("color").recommended_values)
                danger_car_blueprint.set_attribute("color", color)
            if danger_car_blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    danger_car_blueprint.get_attribute("driver_id").recommended_values
                )
                danger_car_blueprint.set_attribute("driver_id", driver_id)

            batch.append(
                SpawnActor(danger_car_blueprint, spawn_point).then(
                    SetAutopilot(FutureActor, True, self.traffic_manager.get_port())
                )
            )

        for i, response in enumerate(self.client.apply_batch_sync(batch, True)):
            if response.error:
                logging.error(response.error)
            else:
                actor_id = response.actor_id
                actor_list = self.world.get_actors()
                danger_car = actor_list.find(actor_id)

                danger_car.set_target_velocity(carla.Vector3D(20))
                self.traffic_manager.set_path(danger_car, route_batch[i])
                self.traffic_manager.ignore_lights_percentage(danger_car, 100)
                self.traffic_manager.ignore_signs_percentage(danger_car, 100)
                self.traffic_manager.ignore_vehicles_percentage(danger_car, 100)
                self.traffic_manager.ignore_walkers_percentage(danger_car, 100)
                self.traffic_manager.random_left_lanechange_percentage(danger_car, 0)
                self.traffic_manager.random_right_lanechange_percentage(danger_car, 0)

                self.danger_cars_list.append(actor_id)

    def destroy(self):
        # Destroy all actors
        if len(self.danger_cars_list) != 0:
            print("\nDestroying %d danger vehicles" % len(self.danger_cars_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.danger_cars_list])
            self.danger_cars_list = []
            self.danger_car_route = []

        if len(self.cars_list) != 0:
            print("\nDestroying %d vehicles" % len(self.cars_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.cars_list])
            self.cars_list = []

        if len(self.walkers_list) != 0:
            for i in range(0, len(self.all_id), 2):
                self.all_actors[i].stop()
            print("\nDestroying %d walkers" % len(self.walkers_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
            self.walkers_list = []
            self.all_id = []
            self.all_actors = []

    def set_traffic_lights_state(self):
        hero_traffic_light = None
        opposite_traffic_light = None

        spawn_points = self.world.get_map().get_spawn_points()

        hero_transform = spawn_points[self.hero_spawn_point]
        opposite_transform = spawn_points[self.opposite_spawn_point]

        actors = self.world.get_actors()
        for actor in actors:
            if isinstance(actor, carla.TrafficLight):
                waypoints = actor.get_stop_waypoints()
                for waypoint in waypoints:
                    if (hero_traffic_light is not None) and (opposite_traffic_light is not None):
                        break
                    tl_trasform = waypoint.transform
                    hero_distance = float(
                        np.sqrt(
                            np.square(hero_transform.location.x - tl_trasform.location.x)
                            + np.square(hero_transform.location.y - tl_trasform.location.y)
                        )
                    )
                    opposite_distance = float(
                        np.sqrt(
                            np.square(opposite_transform.location.x - tl_trasform.location.x)
                            + np.square(opposite_transform.location.y - tl_trasform.location.y)
                        )
                    )
                    min_distance = 10
                    if hero_distance < min_distance:
                        hero_traffic_light = actor
                    if opposite_distance < min_distance:
                        opposite_traffic_light = actor

        if (hero_traffic_light is not None) and (opposite_traffic_light is not None):
            if hero_traffic_light.get_state() == carla.TrafficLightState.Red:
                hero_traffic_light.set_state(carla.TrafficLightState.Green)
                hero_traffic_light.set_green_time(1000.0)
            if opposite_traffic_light.get_state() == carla.TrafficLightState.Red:
                opposite_traffic_light.set_state(carla.TrafficLightState.Green)
                opposite_traffic_light.set_green_time(1000.0)

        for actor in actors:
            if isinstance(actor, carla.TrafficLight):
                if actor != hero_traffic_light:
                    if actor.get_state() == carla.TrafficLightState.Green:
                        actor.set_state(carla.TrafficLightState.Red)
                        actor.set_red_time(1000.0)
                else:
                    continue

    def toggle_traffic_lights(self, unsignalized=True):
        traffic_lights = self.world.get_environment_objects(carla.CityObjectLabel.TrafficLight)
        traffic_signs = self.world.get_environment_objects(carla.CityObjectLabel.TrafficSigns)

        traffic_lights_set = set()
        traffic_signs_set = set()

        for i, traffic_light in enumerate(traffic_lights):
            traffic_lights_set.add(traffic_lights[i].id)

        for i, traffic_sign in enumerate(traffic_signs):
            traffic_signs_set.add(traffic_signs[i].id)

        if unsignalized:
            self.world.enable_environment_objects(traffic_lights_set, False)
            self.world.enable_environment_objects(traffic_signs_set, False)
        else:
            self.world.enable_environment_objects(traffic_lights_set, True)
            self.world.enable_environment_objects(traffic_signs_set, True)
