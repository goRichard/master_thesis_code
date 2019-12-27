from DBSCAN import *

# plot the result
# plot the curve
# plot the curve of neighbouring method with mms
"""
plt.figure()
plt.title("K Neighbouring Distance with data pre-processing mms")
for i in range(5):
    plt.subplot(5, 1, i+1)
    plt.plot(results_neighbouring_mms[4*i+0], "r-",label="min sample = {}".format(4*i+1))
    plt.plot(results_neighbouring_mms[4*i+1], "b-",label="min sample = {}".format(4*i+2))
    plt.plot(results_neighbouring_mms[4*i+2], "g-",label="min sample = {}".format(4*i+3))
    plt.plot(results_neighbouring_mms[4*i+3], "y-", label="min sample = {}".format(4*i+4))
    plt.xlabel("Samples")
    plt.ylabel("K Distance")
    plt.legend(loc="upper left")
plt.savefig("find_optimal_eps_elbow_method_mms.png")
plt.show()

"""




# plot the curve of neighbouring method with ss

plt.figure()
plt.title("K Neighbouring Distance with data pre-processing ss")
for i in range(5):
    plt.subplot(5, 1, i+1)
    plt.plot(results_neighbouring_ss[4*i+0], "r-",label="min sample = {}".format(4*i+1))
    plt.plot(results_neighbouring_ss[4*i+1], "b-",label="min sample = {}".format(4*i+2))
    plt.plot(results_neighbouring_ss[4*i+2], "g-",label="min sample = {}".format(4*i+3))
    plt.plot(results_neighbouring_ss[4*i+3], "y-", label="min sample = {}".format(4*i+4))
    plt.xlabel("Samples")
    plt.ylabel("K Distance")
    plt.legend(loc="upper left")
plt.savefig("find_optimal_eps_elbow_method_ss.png")
plt.show()




# plot the curve of formular method with ss
"""
plt.figure()
plt.title("Results of formular method")
plt.plot(results_formular, "r")
plt.xlabel("min_samples")
plt.ylabel("eps")
plt.savefig("find_optimal_eps_elbow_method_ss.png")
plt.show()

"""

# plot the histogram
# with k gets larger, the distance between core points and neighbours will also gets farther, the threshold would be
# where the distance has a large jump

# plot the hist of eps with neighbouring_mms
"""
plt.figure()
i = 1
for result in results_neighbouring_mms:
    plt.subplot(5, 2, i)
    plt.title("Results of neighbouring method with k = {}".format(i))
    plt.hist(result, bins=20)
    plt.xlabel("K Distance")
    plt.ylabel("number of data samples")
    i += 1
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=2,
                    wspace=0.2)
plt.show()

"""

# plot the histogram
# plot the hist of eps with neighbouring_ss

"""
plt.figure()
i = 1
for result in results_neighbouring_ss:
    plt.subplot(5, 2, i)
    plt.title("Results of neighbouring method with k = {}".format(i))
    plt.hist(result, bins=20)
    plt.xlabel("K Distance")
    plt.ylabel("number of data samples")
    i += 1
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=2,
                    wspace=0.2)
plt.show()

"""


# plot the result
# plot the hist
# plot the hist of min_samples with ss
"""
plt.figure()
i = 1
j = 0
for result in within_eps_list_ss:
    plt.subplot(3, 2, i)
    plt.title("Results of eps = {}".format(eps_list_ss[j]))
    plt.hist(result, bins=10)
    # how many data samples we have when we set eps under between this range
    plt.xlabel("min_samples")
    plt.ylabel("number of data samples")
    i += 1
    j += 1

plt.show()

"""




# plot the hist of min_samples with mms
"""
plt.figure()
i = 1
j = 0
for result in within_eps_list_mss:
    plt.subplot(3, 2, i)
    plt.title("Results of eps = {}".format(eps_list_mss[j]))
    plt.hist(result, bins=10)
    plt.xlabel("min_samples")
    plt.ylabel("number of data samples")
    i += 1
    j += 1

plt.show()


"""
