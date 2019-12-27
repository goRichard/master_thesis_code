import pickle
import time
import matplotlib.pyplot as plt
import wntr
import wntr.network.controls as controls
from MultipliersOperation import *
from WaterNetWorkBasics import WaterNetWorkBasics
import os


def add_control_dictionary():
    control_dictionary = dict()
    keys = input("enter the name of controls (split by spacing)")  # control_1 control2 etc..
    values = input("enter the control contents w.r.t the controls (split by spacing)")
    # control_1,1,P11,>,6.3 control_2,0,P12,<,6 etc..
    dict_keys = keys.split(" ")
    dict_values = values.split(" ")
    for i in range(len(dict_values)):
        dict_values[i] = dict_values[i].split(",")

    for j in range(len(dict_keys)):
        control_dictionary[dict_keys[j]] = dict_values[j]

    return control_dictionary


class WaterNetWorkSimulation(WaterNetWorkBasics):

    def __init__(self, inp_file, town_name, durations):
        assert isinstance(durations, list)
        super(WaterNetWorkSimulation, self).__init__(inp_file, town_name)
        self._durations = durations

    def check_durations(self):
        time_gap = self.wn_time_options['duration'] / self.wn_time_options['pattern_time_step']
        if sum(self._durations) > time_gap:
            print("durations out of range, re-enter the durations")
            return False
        else:
            pass

    @property
    def do_simulation_from_beginning_wntr_1(self):
        """
        use the save and reload method
        """
        print('simulation starts...')
        self.save_simulation_file()
        start = time.time()
        sim = wntr.sim.WNTRSimulator(self._wn, mode="DD")  # EPANETSimulator
        result_wntr_from_beginning = sim.run_sim()
        end = time.time()
        self.reload_simulation_file()
        print('simulation is completed, it takes: {} s'.format(round(end - start, 2)))
        return result_wntr_from_beginning

    @property
    def do_simulation_from_beginning_wntr_2(self):
        # don not use this method
        """
        use the reset initial values method
        """
        start = time.time()
        sim = wntr.sim.WNTRSimulator(self._wn, mode="DD")  # EPANETSimulator
        result_wntr_from_beginning = sim.run_sim()
        end = time.time()
        print('simulation is completed, it takes: {} s'.format(round(end - start, 2)))
        self._wn.reset_initial_values()
        return result_wntr_from_beginning

    @property
    def do_simulation_from_beginning_epanet(self):
        print('simulation starts...')
        self.save_simulation_file()
        start = time.time()
        sim = wntr.sim.EpanetSimulator(self._wn)
        result_epanet_from_beginning = sim.run_sim()
        end = time.time()
        self.reload_simulation_file()
        print("simulation run by EPANET Simulator is completed, it takes: {} s".format(round(end - start, 2)))
        return result_epanet_from_beginning

    @property
    def do_simulation_wntr(self):
        print('simulation starts...')
        start = time.time()
        sim = wntr.sim.WNTRSimulator(self._wn, mode="DD")
        result_wntr = sim.run_sim()
        end = time.time()
        print('simulation is completed, it takes: {} s'.format(round(end - start, 2)))
        return result_wntr

    def save_simulation_file(self):
        f = open('wn.pickle', 'wb')
        pickle.dump(self._wn, f)
        f.close()

    def reload_simulation_file(self):
        f = open('wn.pickle', 'rb')
        self._wn = pickle.load(f)
        f.close()

    def remove_controls(self, control_names):
        for control_name in control_names:
            self._wn.remove_control(control_name)

    def add_controls(self, control_dictionary):
        """
        : type: **kwargs: dictionary
            : keys  : string: name of controls
            : items : list
                : type: link_name: string
                : type: open_or_close: int, 1 or 0, 1 for open and 0 for close
                : type: node_name: string, the name of a node influenced by condition
                : type: above_or_below: string, either larger '>' or smaller '<'
                : type: value: int, above what level and below what level
        """
        # first to check if the control_name is already exist in the current
        # control name list, if it exists, remove it
        assert isinstance(control_dictionary, dict)
        add_controls_names = list(control_dictionary.keys())
        for add_controls_name in add_controls_names:
            if add_controls_name in self.name_list["control_name_list"]:
                self.remove_controls(add_controls_name)
                # print("{} has been removed".format(add_controls_name))

            # set the actions of certain link

            control_link = self.get_link(control_dictionary[add_controls_name][0])
            control_act = controls.ControlAction(control_link, 'status', int(control_dictionary[add_controls_name][1]))

            # set the conditions regarding the control link

            control_node = self.get_node(control_dictionary[add_controls_name][2])
            control_cond = controls.ValueCondition(control_node, 'level', control_dictionary[add_controls_name][3],
                                                   float(control_dictionary[add_controls_name][4]))

            new_control = controls.Control(control_cond, control_act)
            self._wn.add_control(add_controls_name, new_control)
        print("add controls completed")

    def change_multipliers(self, pattern_name):
        # get the operation list
        op_list = operation_list()
        # get the multipliers
        multipliers = self.get_pattern_multipliers()[pattern_name]
        # set the changed multipliers
        changed_multipliers = []
        for i in range(len(multipliers)):
            operation = multiplier_operation(op_list, up_prob=0.1, down_prob=0.2, left_prob=0.2,
                                             right_prob=0.2, gaussian_prob=0.3)

            if operation in op_list[:2]:
                changed_multipliers.append(operation(multipliers, i))
            else:
                changed_multipliers.append(operation(multipliers[i]))
            print('the operation {} has been chose'.format(operation.__name__))
        return changed_multipliers

    def compare_multipliers(self, pattern_name, changed_multipliers, original_multipliers):
        """
        但是请注意，参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数 #
        (self, var1 = true, var2 = false, var3, var4, *args, **kwargs)。
        """
        x = self.get_time_steps
        fig1 = plt.figure()
        y1, = plt.plot(x, changed_multipliers[:len(x)], 'r')
        y2, = plt.plot(x, original_multipliers[:len(x)], 'b')
        plt.title("{} changed curve".format(pattern_name))
        plt.xlabel("time(h)")
        plt.ylabel("Demand(L/s)")
        plt.legend([y2, y1], ["changed", "original"], loc='upper left')
        plt.grid(True)
        plt.draw()
        plt.pause(6)
        plt.close(fig1)

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

        sim_results = []
        self.check_durations()
        for i in range(len(self._durations)):
            self._wn.options.time.duration = sum(self._durations[:i + 1]) * 3600
            print('\n')
            print("new simulation starts")
            print('-' * 50)
            add_controls = input('add controls? (True/False): ')
            if add_controls == 'True':
                control_dictionary = add_control_dictionary()
                self.add_controls(control_dictionary)
            print('-' * 50)
            change_multipliers_or_not = input('change the pattern multipliers? (True or False)')
            if change_multipliers_or_not == 'True':
                pattern_names = input('name of the pattern multipliers (split by spacing): ').split(' ')
                for pattern_name in pattern_names:
                    original_multipliers = self.get_pattern_multipliers()[pattern_name]
                    changed_multipliers = self.change_multipliers(pattern_name)
                    print("multipliers changed done")
                    self.compare_multipliers(pattern_name, changed_multipliers,
                                             original_multipliers)
                    self._wn.get_pattern(pattern_name).multipliers = changed_multipliers
                print("-" * 50)

            restart_or_not = input('reset the simulation time after this simulation? (True/False)')
            print('-' * 50)
            if restart_or_not == 'True':
                epanet_or_wntr = input("use EPANET or WNTR simulator: ")
                if epanet_or_wntr == "EPANET":
                    sim_results.append(self.do_simulation_from_beginning_epanet)
                    print("simulation duration from {} h to {} h ".format(0, sum(self._durations[:i + 1])))
                else:
                    # self._wn.options.time.duration = sum(self._durations[:i + 1]) * 3600
                    sim_results.append(self.do_simulation_from_beginning_wntr_1)
                    print("simulation duration from {} h to {} h ".format(0, sum(self._durations[:i + 1])))
                    # sim_results.append(self.do_simulation_from_beginning_wntr_2) don not use this method
            else:
                # self._wn.options.time.duration = sum(self._durations[:i + 1]) * 3600
                sim_results.append(self.do_simulation_wntr)
                print("simulation duration from {} h to {} h ".format(sum(self._durations[:i]),
                                                                      sum(self._durations[:i + 1])))
            print("-" * 50)

            remove_controls = input('remove controls? (True/False):')
            if remove_controls == 'True':
                remove_control_list = input('control to be removed (dictionary type):')
                self.remove_controls(remove_control_list)
            print("-" * 50)
        return sim_results


if __name__ == "__main__":
    inp_file = "data/c-town_true_network.inp"
    c_town_simulation = WaterNetWorkSimulation(inp_file, "c_town", [168])
    results = c_town_simulation.iterate_simulation()
    save_path = os.getcwd() + "/data"
    print("save file at : {}".format(save_path))
    i = 0
    for result in results:
        file_name = save_path + '/' + 'pressure_' + str(i) + '_epanet_whole' + '.csv'
        #        file_name_disturbance = current_path + '/' + 'disturbance_' + str(
        #           i) + '_multipliers_changed_down_dma_5' + '.csv'
        #        file_name_flow_rate = current_path + '/' + 'flow_rate_' + str(
        #         i) + '_multipliers_changed_down_dma_5' + '.csv'

        result.node['pressure'].to_csv(file_name)
        #        result.node['demand'].to_csv(file_name_disturbance)
        #        result.link['flowrate'].to_csv(file_name_flow_rate)
        i += 1
