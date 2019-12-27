#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:08:27 2019

@author: ruizhiluo
"""

import pickle
import time
import matplotlib.pyplot as plt
import wntr
import wntr.network.controls as controls
import os




class TestWN:
    """
    this class contains attributes and methods to test the water network of
    c_town
    """

    def __init__(self, file_path):
        """
        constructor
        : type file_path: string
        """
        assert isinstance(file_path, str)
        self._file_path = file_path
        self.wn = wntr.network.model.WaterNetworkModel(self._file_path)
        # self.timeSim = timeSim

    def get_node_name_list(self):
        """
        get the node name of .inp file
        : rtype: list contains three different node types
        """
        node_names = self.wn.node_name_list
        tank_names = [tank for tank in node_names if tank.startswith("T")]
        reservoir_names = [reservoir for reservoir in node_names if reservoir.startswith("R")]
        junction_names = [junction for junction in node_names if junction.startswith("J")]
        return [tank_names, reservoir_names, junction_names]

    def get_link_name_list(self):
        """
        get the link name of .ipn file
        : rtype: list contains different link types

        """

        link_names = self.wn.link_name_list
        pump_names = [pump for pump in link_names if pump.startswith("PU")]
        pipe_names = [pipe for pipe in link_names if pipe.startswith("P")]
        valve_names = [valve for valve in link_names if valve.startswith("V")]

        return [pump_names, pipe_names, valve_names]

    def get_pattern_multipliers(self, pattern_name):
        """
        get the specific demand pattern
        demand pattern unit (L/s)
        """
        multipliers = self.wn.get_pattern(pattern_name).multipliers
        return multipliers

    def get_time_range(self):
        """
        the whole time range of of .inp file
        not the simulation time range
        """
        duration = self.wn.options.time.duration
        pattern_time_step = self.wn.options.time.pattern_timestep
        pattern_start = self.wn.options.time.pattern_start
        time_series = np.arange(pattern_start, duration + pattern_time_step, pattern_time_step)

        return time_series

    def get_control_name_list(self):
        """
        get the name of all control
        """
        return self.wn.control_name_list

    def get_controls(self, control_name):
        """
        get the handle of a specific control:
        it look like:
        <Control: 'control2', <ValueCondition: T1, level, >=, 6.3>,
        [<ControlAction: PU1, status, Closed>],
        [], priority=3>
        example of control 2

        it will return a object control from class wntr.network.controls
        """
        control = self.wn.get_control(control_name)
        return control

    def get_junctions(self, junction_name):
        junction = self.wn.get_node(junction_name)
        return junction

    def get_pumps(self, pump_name):
        """

        :param pump_name: string
        :return: the object of pump in class wntr.network.elements.Pump
        the class wntr.network.elements.Pump has property like:
            start_node
            end_node
            status
            etc.
        """
        pump = self.wn.get_link(pump_name)
        return pump

    def get_valve(self, valve_name):
        valve = self.wn.get_link(valve_name)
        return valve


class Simulation(TestWN):

    def __init__(self, file_path, durations):
        """

        :type file_path: str
        """
        super().__init__(file_path)
        assert isinstance(file_path, str)
        self._file_path = file_path
        self.wn = wntr.network.model.WaterNetworkModel(self._file_path)
        assert isinstance(durations, list)
        self.durations = durations

    def check_duration(self):
        if sum(self.durations) * 3600 > self.wn.options.time.duration:
            print("please re-enter the time durations")
            return False
        else:
            return True


    def plot_changed_pattern_multipliers(self, pattern_name, changed_multipliers, original_multipliers):
        """
        但是请注意，参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数 #
        (self, var1 = true, var2 = false, var3, var4, *args, **kwargs)。
        """

        y_1 = np.array([multiplier for multiplier in changed_multipliers]).reshape(-1, 1)
        y_2 = np.array([multiplier for multiplier in original_multipliers]).reshape(-1, 1)
        x = np.linspace(0, y_1.shape[0], y_1.shape[0] + 1)[1:].reshape(-1, 1)
        plt.figure()
        y1, = plt.plot(x, y_1, 'r')
        y2, = plt.plot(x, y_2, 'b')
        plt.title("{} changed curve".format(pattern_name))
        plt.xlabel("time(h)")
        plt.ylabel("Demand(L/s)")
        plt.legend([y2, y1], ["changed", "original"], loc='upper left')
        plt.show()

    def iterate_simulation(self):

        """
        set mode to DD: demand driven
        : type: list, simulation time
        : type: bool, check if the simulation needs to be restarted
        : type: int, number of time split
        : rtype: WNTR simulation, it contains:
            1. Timestamp
            2. Network Name
            3. Node Results
            4. Link Results

            Link and Node Results are both dictionaries of PANDA dataFrame

        """
        if self.check_duration():
            sim_results = []
            for i in range(len(self.durations)):
                self.wn.options.time.duration = sum(self.durations[:i + 1]) * 3600
                print('*******************************')
                add_controls = input('add controls? (True/False): ')
                if add_controls == 'True':
                    add_control_dict = input('control to be added (dictionary type):')
                    self.add_controls(**add_control_dict)

                change_multipliers_or_not = input('change the pattern multipliers?')
                if change_multipliers_or_not == 'True':
                    pattern_names = input('name of the pattern multipliers: ').split(' ')
                    for pattern_name in pattern_names:
                        original_multipliers = self.wn.get_pattern(pattern_name).multipliers
                        # self.plot_changed_pattern_multipliers(pattern_name,
                        #                                      original_multipliers)

                        # self.wn.get_pattern(pattern_name).multipliers = [multiplier + float(np.random.random(1)) for multiplier in
                        #                                                 self.wn.get_pattern(pattern_name).multipliers]
                        original_multipliers = self.wn.get_pattern(pattern_name).multipliers
                        op_list = self.multipliers_operation_list()
                        operation = self.multipliers_operation_random_choose(op_list)
                        if operation in op_list[:4]:
                            self.wn.get_pattern(pattern_name).multipliers = [operation(multiplier) for
                                                                             multiplier in original_multipliers]
                        else:
                            self.wn.get_pattern(pattern_name).multipliers = operation(
                                self.wn.get_pattern(pattern_name).multipliers)


                        print('the operation {} has been chose'.format(operation.__name__))

                        self.plot_changed_pattern_multipliers(pattern_name,
                                                              self.wn.get_pattern(pattern_name).multipliers,
                                                              original_multipliers)
                        """
                        operation = self.multipliers_operation_random_choose()
                        self.wn.get_pattern(pattern_name).multipliers = \
                            [operation(multiplier)
                             for multiplier in original_multipliers]
                       
                        self.plot_changed_pattern_multipliers(pattern_name,
                                                              self.wn.get_pattern(pattern_name).multipliers)
                        
                        """

                        # np.random.random will return random floats in the half-open interval [0.0, 1.0). (from
                        # continuous uniform distribution)

                        # np.random.randn will return random floats from the standard normal distribution X ~ N(0,
                        # 1) change the mean and variance by adding value (change mean) and multiply value (change
                        # variance)
                        # sigma * np.random.randn(...) + mu
                        # e.g.  X = np.random.randn() ~ N(0,1) then:
                        # np.sqrt(0.1) * X + 2 ~ N(2, 0.1)

                restart_or_not = input('reset the simulation time after this simulation? (True/False)')
                if restart_or_not == 'True':
                    sim_results.append(self.do_simulation_from_beginning())
                else:
                    sim_results.append(self.do_simulation())
                remove_controls = input('remove controls? (True/False):')
                if remove_controls == 'True':
                    remove_control_list = input('control to be removed (dictionary type):')
                    self.remove_controls(*remove_control_list)

        else:
            return 0
        self.wn.reset_initial_values()
        return sim_results

    def do_simulation_from_beginning(self):
        self.save_simulation_file()
        start = time.time()
        sim = wntr.sim.WNTRSimulator(self.wn, mode="DD") # EPANETSimulator
        result_0 = sim.run_sim()
        end = time.time()
        self.reload_simulation_file()
        print('simulation is completed, it takes: {} s'.format(round(end - start, 2)))
        return result_0

    def do_simulation(self):
        start = time.time()
        sim = wntr.sim.WNTRSimulator(self.wn, mode="DD")
        result_1 = sim.run_sim()
        end = time.time()
        print('simulation is completed, it takes: {} s'.format(round(end - start, 2)))
        return result_1

    def remove_controls(self, *args):
        for control_name in args:
            self.wn.remove_control(control_name)

    def add_controls(self, **kwargs):
        """
        : type: **kwargs: dictionary
            : keys  : string: name of controls
            : items : list
                : type: link_name: string
                : type: open_or_close: 1 or 0, 1 for open and 0 for close
                : type: node_name: string, the name of a node influenced by condition
                : type: above_or_below: string, either larger '>' or smaller '<'
                : type: value: int, above what level and below what level
        """
        # first to check if the control_name is already exist in the current
        # control name list, if it exists, remove it

        control_name_list = self.wn.control_name_list
        for control_name in list(kwargs.keys()):
            if control_name in control_name_list:
                self.wn.remove_control(control_name)

            # set the actions of certain link

            control_link = self.wn.get_link(kwargs[control_name][0])
            control_act = controls.ControlAction(control_link, 'status', kwargs[control_name][1])

            # set the conditions regarding the control link

            control_node = self.wn.get_node(kwargs[control_name][2])
            control_cond = controls.ValueCondition(control_node, 'level', kwargs[control_name][3],
                                                   kwargs[control_name][4])

            new_control = controls.Control(control_cond, control_act)
            self.wn.add_control(control_name, new_control)

    def save_simulation_file(self):
        f = open('wn.pickle', 'wb')
        pickle.dump(self.wn, f)
        f.close()

    def reload_simulation_file(self):
        f = open('wn.pickle', 'rb')
        self.wn = pickle.load(f)
        f.close()

    def get_simulation_pressure(self):
        """
        : type: *args, list, the simulation result from WNTR simulator
        : type: which_duration, int, the arrangement of durations
        returns a pandas dataframe
        """
        pressure = []
        for sim_result in self.sim_results:
            pressure.append(sim_result.node['pressure'])

        return pressure

    def plot_simulation_pressure(self, node_name, duration_period):
        # if node_name in self.get_node_name():
        pressure = self.get_simulation_pressure()
        node_pressure = pressure[duration_period][node_name].values
        simulation_time = np.array(pressure[duration_period][node_name].index) / 3600
        plt.figure(figsize=(8, 4))
        plt.plot(simulation_time, node_pressure)
        plt.title('{} \'s pressure in {} hours'.format(node_name, self.durations[duration_period]))
        plt.xlabel('time(h)')
        plt.ylabel('pressure (m)')
        plt.grid(True)
        plt.show()

    def simulation_result_node(self, *args):
        """
        get the results of node
        : type: node_name, string
        : rtype: dictionary
            (key-value pair)
            Keys contain Demand, Leak Demand, Pressure and Head
        """

        node_dict = self.sim_results.node

        '''
        : type: pandas data frame, contains timestamp and values
            values of Demand, Leak Demand, Pressure and Head
        '''
        return node_dict

    @staticmethod
    def simulation_results_link(*args):
        """
        : rtype:dictionary
            (key-value pair)
            1. a dictionary (key-value pair)
            2. Keys contain flowrate, velocity and status
            3. status 0 : closed
               status 1 : open
        """

        link_dict = [arg.link for arg in args]
        return link_dict

    def baseline(self):
        generator = self.wn.junctions()
        baseline = []
        for j_name, j_object in generator:
            baseline.append(j_object.demand_timeseries_list[0].base_value)

        baseline = np.array(baseline).reshape(1, -1)
        return baseline


class PlotWn(TestWN):

    def __init__(self, _file_path):
        super(PlotWn, self).__init__(_file_path)
        self._file_path = _file_path
        self.wn = wntr.network.model.WaterNetworkModel(self._file_path)

    # following methods will plot the location of junction, tank, reservoir or pump etc. in specific town
    def plot_junction(self, title):
        node_attribute = self.get_node_name_list()
        wntr.graphics.plot_network(self.wn, node_attribute[2], None, title, node_size=20)

    def plot_tank(self, title):
        node_attribute = self.get_node_name_list()
        wntr.graphics.plot_network(self.wn, node_attribute[0], None, title)

    def plot_reservoir(self, title):
        node_attribute = self.get_node_name_list()
        wntr.graphics.plot_network(self.wn, node_attribute[1], None, title)

    def plot_valve(self, title):
        link_attribute = self.get_link_name_list()
        wntr.graphics.plot_network(self.wn, None, link_attribute[2], title)

    def plot_pump(self, title):
        link_attribute = self.get_link_name_list()
        wntr.graphics.plot_network(self.wn, None, link_attribute[0], title)
        plt.show()

    def plot_pipe(self, title):
        link_attribute = self.get_link_name_list()
        wntr.graphics.plot_network(self.wn, None, link_attribute[1], title)

    def plot_pump_curve(self, pump_name):
        """
        plot the char. curve of pump
        in c town 11 pumps
        """

        pump = self.wn.get_link(pump_name)
        wntr.graphics.curve.plot_pump_curve(pump, title=pump_name)
        plt.show()

    def plot_demand_pattern(self, pattern_name):
        """
        plot a specific patter in a certain simulation time
        """
        multipliers = self.get_pattern_multipliers(pattern_name)
        x = self.get_time_range()[1:].reshape(-1, 1) / 3600
        y = multipliers.reshape(-1, 1)
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title("{} Curve".format(pattern_name))
        plt.xlabel("time(h)")
        plt.ylabel("Demand (L/s)")
        plt.grid(True)
        plt.show()

    def plot_junction_pattern(self, junction_name):
        """
        : type: junction_name, string
        plot the pattern of a specific junction
        """
        # check if this junction has no pattern
        junction = self.wn.get_node(junction_name)
        if junction.demand_timeseries_list[0].pattern is not None:
            pattern = junction.demand_timeseries_list[0].pattern
            time_series = self.get_time_range()
            plt.figure()
            plt.title('{} \'s demand pattern '.format(junction_name))
            plt.plot(time_series, pattern, 'r-')
            plt.xlabel('time(h)')
            plt.ylabel('Demand(L/s)')

    def plot_changed_pattern_multipliers(self, pattern_name, *args):
        """
        但是请注意，参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数 #
        (self, var1 = true, var2 = false, var3, var4, *args, **kwargs)。
        """
        x = self.get_time_range()[1:].reshape(-1, 1) / 3600
        y = np.array([arg for arg in args]).reshape(-1, 1)
        plt.figure()
        plt.plot(x, y)
        plt.title("{} changed curve".format(pattern_name))
        plt.xlabel("time(h)")
        plt.ylabel("Demand(L/s)")

    # these two methods only plot after the simulation
    def plot_node(self, node_key, node_name):
        node_dict = self.simResultNode()
        y = node_dict[node_key].loc[:, node_name].values.reshape(-1, 1)  ## pressure, demand, head or leak demand
        x = node_dict[node_key].index.values.reshape(-1, 1) / 3600
        plt.figure()
        plt.plot(x, y)
        plt.title("{} at {}".format(node_key, node_name))
        plt.xlabel("time(h)")
        plt.ylabel(node_key)
        plt.show()

    def plot_link(self, link_key, link_name):
        link_dict = self.simResultLink()
        y = link_dict[link_key].loc[:, link_name].values.reshape(-1, 1)  ## flowrate, velocity or status
        x = link_dict[link_key].index.values.reshape(-1, 1) / 3600
        plt.figure()
        plt.plot(x, y)
        plt.title("{} at {}".format(link_key, link_name))
        plt.xlabel("time(h)")
        plt.ylabel(link_name)
        plt.show()


if __name__ == '__main__':
    inp_file = "data/c-town_true_network.inp"
    c_town = TestWN(inp_file)
    c_town_plot = PlotWn(inp_file)
    c_town_sim = Simulation(inp_file, [40, 40, 40, 48])
    results = c_town_sim.iterate_simulation()
    current_path = os.getcwd()
    # results.node['head'].to_csv('changed_gaussian.csv')
    i = 0
    # c_town_plot.plot_pump_curve("PU1")
    # c_town_plot.plot_pump_curve("PU2")
    # plt.show()
    # PU1 = c_town.get_pumps("PU1")
    # print(PU1.flow)
    # for result in results:
    #    file_name_head = current_path + '/' + 'head_' + str(i) + '_multipliers_changed_down_dma_5' + '.csv'
    #    file_name_disturbance = current_path + '/' + 'disturbance_' + str(
    #        i) + '_multipliers_changed_down_dma_5' + '.csv'
    #    file_name_flow_rate = current_path + '/' + 'flow_rate_' + str(
    #        i) + '_multipliers_changed_down_dma_5' + '.csv'

    #    result.node['head'].to_csv(file_name_head)
    #    result.node['demand'].to_csv(file_name_disturbance)
    #    result.link['flowrate'].to_csv(file_name_flow_rate)
    #    i += 1
