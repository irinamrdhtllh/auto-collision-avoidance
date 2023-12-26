class Traffic:
    def __init__(self, core, exp_config):
        self.core = core
        self.exp_config = exp_config
        self.crossroad = self.exp_config["crossroad_type"]

    def background_traffic(self):
        raise NotImplementedError


class Low(Traffic):
    def __init__(self, core, exp_config):
        super().__init__(core, exp_config)


    def background_traffic(self):
        self.core.destroy()
        self.core.reset_hero()
        self.core.spawn_danger_car()

        if self.crossroad == "unsignalized":
            self.core.toggle_traffic_lights()
        else:
            self.core.set_traffic_lights_state()


class Medium(Traffic):
    def __init__(self, core, exp_config):
        super().__init__(core, exp_config)

    def background_traffic(self):
        self.core.destroy()
        self.core.reset_hero()
        self.core.spawn_danger_car()
        self.core.spawn_cars()

        if self.crossroad == "unsignalized":
            self.core.toggle_traffic_lights()
        else:
            self.core.set_traffic_lights_state()


class High(Traffic):
    def __init__(self, core, exp_config):
        super().__init__(core, exp_config)

    def background_traffic(self):
        self.core.destroy()
        self.core.reset_hero()
        self.core.spawn_danger_car()
        self.core.spawn_cars()
        self.core.spawn_walkers()

        if self.crossroad == "unsignalized":
            self.core.toggle_traffic_lights()
        else:
            self.core.set_traffic_lights_state()
