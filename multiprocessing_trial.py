import multiprocessing

def worker(num):
    print("worker: {}".format(num))


if __name__ == "__main__":
    jobs = []
    for i in range(10):
        p = multiprocessing.Process(target=worker, args=(i, ))
        jobs.append(p)
        p.start()
