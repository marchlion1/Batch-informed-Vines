import time


class Timer:
    def __init__(self):
        self.end_timers = {}
        self.counters = {}
        self.st_time = time.time()

    def stop(self, name):
        if name not in self.end_timers:
            self.end_timers[name] = time.time()

    def count_operation(self, name):
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += 1

    def show(self):
        for name, count in self.counters.items():
            print(f"{name} - Number of operations: {count}")
        for name, ed_time in self.end_timers.items():
            print(str(name) + ": spend " + str(ed_time - self.st_time))


# Example usage

if __name__ == "__main__":
    timer = Timer()
    timer.start("Task 1")
    for i in range(1000000):
        timer.count_operation("Task 1")
    timer.stop("Task 1")
    timer.show()