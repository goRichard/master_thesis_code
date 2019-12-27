import numpy as np

"""
methods to modify the multipliers
"""


def move_multiplier_right(multipliers, index):
    """
    :param index: int
    :param multipliers: list
    :return: multiplier: float
    """
    assert isinstance(multipliers, list)
    right_parameter = np.random.randint(1, len(multipliers))
    if (index + right_parameter + 1) > len(multipliers):
        multiplier = multipliers[index + right_parameter - len(multipliers)]
    else:
        multiplier = multipliers[index + right_parameter]
    return multiplier


def move_multiplier_left(multipliers, index):
    assert isinstance(multipliers, list)
    left_parameter = np.random.randint(1, len(multipliers))
    multiplier = multipliers[index - left_parameter]
    return multiplier


def move_multiplier_right_whole(multipliers):
    """
    :param multipliers: list
    :return:
    """
    assert isinstance(multipliers, list)
    multipliers = []

    right_parameter = np.random.randint(1, 10)
    for i in range(multipliers.shape[0]):
        if i <= right_parameter:
            multipliers.append(multipliers[-i - 1])
            continue
        multipliers.append(multipliers[i - right_parameter - 1])
    print('multipliers have been shifted {} to the right side'.format(right_parameter))
    return multipliers


def move_multipliers_left_whole(multipliers):
    """
    : type: int
    : type: np.array, set to one column
    """
    multipliers = np.array(multipliers).reshape(-1, 1)
    multipliers = np.zeros(multipliers.shape)
    left_parameter = np.random.randint(1, 10)
    for i in range(multipliers.shape[0]):
        if i <= left_parameter:
            multipliers[-i - 1] = multipliers[i]
            continue
        multipliers[i - left_parameter - 1] = multipliers[i, 0]
    print('multipliers have been shifted {} to the left side'.format(left_parameter))
    return multipliers


def move_multiplier_up(multiplier):
    """
    :param multiplier: float: the original multiplier
    :return: the modified multiplier
    """
    up_parameter = float(np.random.uniform(0, 0.1, 1))  # pick a parameter randomly in uniform (0, 0.1)
    multiplier += up_parameter
    return multiplier


def move_multiplier_down(multiplier):
    """
    move a single multiplier down, if the it becomes negative then set it to zero
    :param multiplier: float: the original multiplier
    :return: the modified multiplier
    """
    # avoid the multiplier to be negative
    down_parameter = float(np.random.uniform(0, 0.1, 1))

    if (multiplier - down_parameter) < 0:
        multiplier = 0
    else:
        multiplier -= down_parameter
    return multiplier


def gaussian_operation(multiplier):
    """
    :param multiplier: float, single multiplier of one pattern
    :return: the modified multipliers

    this method add a random noise on the original multipliers
    """
    noise = float(np.random.normal(0, 0.1, 1))
    # if the noise is negative and the absolute value is larger than multiplier, then set the multiplier to 0, avoid
    # the multiplier to be negative
    if noise < 0 and abs(noise) > multiplier:
        multiplier = 0
    else:
        multiplier += noise
    return multiplier


def operation_list():
    # get the functions
    op_gaussian = gaussian_operation
    op_move_up = move_multiplier_up
    op_move_down = move_multiplier_down
    op_move_left = move_multiplier_left
    op_move_right = move_multiplier_right
    op_move_all_left = move_multipliers_left_whole
    op_move_all_right = move_multiplier_right_whole

    # define the list
    op_list = [op_move_left, op_move_right, op_move_up, op_move_down, op_gaussian]
    op_list_all = [op_move_all_left, op_move_all_right]

    return op_list


def multiplier_operation(op_list, up_prob=0.2, down_prob=0.2, left_prob=0.2, right_prob=0.2, gaussian_prob=0.2):
    """
    :param op_list: list, operation list
    :param up_prob: float, probability
    :param down_prob: float, probability
    :param left_prob: float, probability
    :param right_prob: float, probability
    :param gaussian_prob: float, probability
    :return: func, chosen function
    """
    prob_distribution = [up_prob, down_prob, left_prob, right_prob, gaussian_prob]
    assert sum(prob_distribution) == 1, "sum of probability distribution should be 1"

    # choose the operation from list with some probability distribution (discrete)
    operation = np.random.choice(op_list, p=prob_distribution)
    return operation
