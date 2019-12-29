import matplotlib.pyplot as plt
import wntr
import numpy as np
import copy


class WaterNetWorkBasics:
    """
    this class creates some attributes and methods for
    a specific water network
    """

    def __init__(self, inp_file, town_name):
        self.inp_file = inp_file
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)
        self.town_name = town_name
        self.time_step = self.wn_time_options["pattern_time_step"]
        self.duration = self.wn_time_options["duration"]
        self.time_gap = int(self.duration / self.time_step)
        self.town_name = town_name

    @property
    def wn_time_options(self):
        time_options = {"duration": self.wn.options.time.duration,
                        "quality_time_step": self.wn.options.time.quality_timestep,
                        "pattern_time_step": self.wn.options.time.pattern_timestep,
                        "pattern_start": self.wn.options.time.pattern_start}
        return time_options

    @property
    def describe(self):
        description = {'junctions': self.wn.num_junctions,
                       'junctions_with_demand': len(self.name_list['node_name_list']['junctions_with_demand']),
                       'tanks': self.wn.num_tanks,
                       'reservoirs': self.wn.num_reservoirs, 'valves': self.wn.num_valves,
                       'pipes': self.wn.num_pipes,
                       'pumps': self.wn.num_pumps, "controls": self.wn.num_controls,
                       "patterns": self.wn.num_patterns}
        return description

    @property
    def name_list(self):
        # node name
        node_name_list = self.wn.node_name_list
        tank_names = [tank for tank in node_name_list if tank.startswith("T")]
        reservoir_names = [reservoir for reservoir in node_name_list if reservoir.startswith("R")]
        junction_names = [junction for junction in node_name_list if junction.startswith("J")]
        junctions_with_demand = []

        for junction_name in junction_names:
            junction = self.get_node(junction_name)
            if junction.demand_timeseries_list[0].pattern is not None:
                junctions_with_demand.append(junction_name)

        # link name
        link_names = self.wn.link_name_list
        pump_names = [pump for pump in link_names if pump.startswith("PU")]
        pipe_names = [pipe for pipe in link_names if pipe.startswith("P")]
        valve_names = [valve for valve in link_names if valve.startswith("V")]

        # name list
        name_list = {"control_name_list": self.wn.control_name_list,
                     "pattern_name_list": self.wn.pattern_name_list,
                     "node_name_list": {"tank_names": tank_names, "reservoir_names": reservoir_names,
                                        "junction_names": junction_names,
                                        "junctions_with_demand": junctions_with_demand},
                     "link_name_list": {"pump_names": pump_names, "pipe_names": pipe_names, "valve_names": valve_names}}
        return name_list

    def get_node(self, node_name):
        """
        :param node_name: str, name of a node (junctions, tanks, reservoirs)
        :return: the object of this node
        """
        assert isinstance(node_name, str), "node name should be string"
        node = self.wn.get_node(node_name)
        return node

    def get_link(self, link_names):
        """
        :param link_name: str, name of a link (pipes, valves, pumps)
        :return:
        """
        link_list = []
        for link_name in link_names:
            link = self.wn.get_link(link_name)
            link_list.append(link)
        return link_list

    def get_control(self, control_name):
        """
        get the handle of a specific control:
        it look like:
        <Control: 'control', <ValueCondition: T1, level, >=, 6.3>,
        [<ControlAction: PU1, status, Closed>],
        [], priority=3>
        example of control 2

        it will return a object control from class wntr.network.controls
        """
        control = {"control name": self.wn.get_control(control_name).name,
                   "control condition": self.wn.get_control(control_name).condition}
        return control

    def get_pattern_multipliers(self):
        """
        get the specific demand pattern
        demand pattern unit (L/s)
        return: multipilers dictionary. keys: name of pattern, values: multipliers
        """
        pattern_names = self.wn.pattern_name_list
        multipliers = dict()
        for pattern_name in pattern_names:
            multipliers[pattern_name] = copy.deepcopy(list(self.wn.get_pattern(pattern_name).multipliers))
        return multipliers

    @property
    def get_time_steps(self):
        time_step = self.wn_time_options["pattern_time_step"]
        duration = self.wn_time_options["duration"]
        time_gap = int(duration / time_step)
        time_steps = list(np.linspace(0, duration, time_gap + 1))
        time_series =[time/3600 for time in time_steps ]
        return time_series


class PlotWaterNetWorks(WaterNetWorkBasics):

    def __init__(self, inp_file, town_name):
        super().__init__(inp_file, town_name)
        self.town_name = town_name

    # following methods will plot the location of junction, tank, reservoir or pump etc. in specific town

    def plot_loc_junctions(self):
        title = "the location of junctions in" + self.town_name
        junction_names = self.name_list["node_name_list"]["junction_names"]
        wntr.graphics.plot_network(self.wn, junction_names, None, title, node_size=20)
        plt.show()

    def plot_loc_junctions_within_same_pressure(self, junction_list):
        """
        plot the locations of junctions which are classified in the same cluster
        :param junction_list:
        :return: plot of locations of junctions within same cluster
        """
        title = "the location of junctions within same cluster"
        plt.ion()
        wntr.graphics.plot_network(self.wn, junction_list, None, title, node_size=20)
        plt.ioff()
        plt.show()

    def plot_loc_tank(self):
        title = "the location tanks in" + self.town_name
        tank_names = self.name_list["node_name_list"]["tank_names"]
        wntr.graphics.plot_network(self.wn, tank_names, None, title)
        plt.show()

    def plot_loc_reservoir(self):
        title = "the location reservoirs in" + self.town_name
        reservoir_names = self.name_list["node_name_list"]["reservoir_names"]
        wntr.graphics.plot_network(self.wn, reservoir_names, None, title)
        plt.show()

    def plot_loc_valve(self):
        title = "the location valves in" + self.town_name
        valve_names = self.name_list["link_name_list"]["valve_names"]
        wntr.graphics.plot_network(self.wn, valve_names, None, title)
        plt.show()

    def plot_loc_pump(self):
        title = "the location pumps in" + self.town_name
        pump_names = self.name_list["link_name_list"]["pump_names"]
        wntr.graphics.plot_network(self.wn, pump_names, None, title)
        plt.show()

    def plot_loc_pipe(self):
        title = "the location pipes in" + self.town_name
        pipe_names = self.name_list["link_name_list"]["pipe_names"]
        pipe_names.remove("P1")
        wntr.graphics.plot_network(self.wn, pipe_names, None, title)
        plt.show()

    def plot_pump_curve(self, pump_name):
        """
        plot the char. curve of pump
        in c town 11 pumps
        """
        title = pump_name + " curve"
        pump = self.get_link(pump_name)
        wntr.graphics.curve.plot_pump_curve(pump, title=title)
        plt.show()

    def plot_demand_pattern(self, pattern_name):
        """
        plot a specific patter in a certain simulation time
        """
        x = self.get_time_steps
        y = self.get_pattern_multipliers(pattern_name)
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title("{} Curve".format(pattern_name))
        plt.xlabel("time(h)")
        plt.ylabel("Demand (L/s)")
        plt.grid(True)
        plt.show()

    def plot_junction_has_pattern(self, junction_name):
        """
        : type: junction_name, string
        plot the pattern of a specific junction
        """
        # check if this junction has no pattern
        # assert junction_name in self.name_list['node_name_list']['junction_has_pattern_names']
        junction = self.get_node(junction_name)
        pattern_name = junction.demand_timeseries_list[0].pattern
        pattern = self.get_pattern_multipliers()[str(pattern_name)]
        time_series = self.get_time_steps
        plt.figure()
        plt.title('{} \'s demand pattern '.format(junction_name))
        plt.plot(time_series, pattern, 'r-')
        plt.xlabel('time(h)')
        plt.ylabel('Demand(L/s)')
        plt.show()


if __name__ == "__main__":
    c_town = WaterNetWorkBasics("data/c-town_true_network.inp", "c_town")
    c_town_plot = PlotWaterNetWorks("data/c-town_true_network.inp", "c_town")
    c_town_plot.plot_loc_junctions()
