from threading import Thread
import os
import json
from worker import Worker

class Swarm:

    def __init__(self, workers_number, test_time, wait_time, image_path, debug=True):
        print(f"Init swarm #={workers_number}")
        self.workers_number = workers_number
        self.test_time = test_time
        self.wait_time = wait_time
        self.debug = debug
        self.image_path = image_path

    def run_experiment(self, interval=1):
        print(f"Running experiment #={self.workers_number}")
        workers = [Worker(
            threadID=i,
            wait_time=self.wait_time,
            test_time=self.test_time,
            experiment=self.workers_number,
            image_path=self.image_path
        ) for i in range(self.workers_number)]
        for worker in workers:
            
            
            worker.start()
        for worker in workers:
            worker.join()
        logs = []
        for worker in workers:
            logs.extend(worker.logs)
        os.makedirs("experiments", exist_ok=True)
        json.dump(
            logs,
            open(f"./experiments/{self.workers_number}-{self.test_time}-{self.wait_time}.json", "w"),
            skipkeys=",",
            indent=4
        )


if __name__ == "__main__":
    for i in range(100):
        swarm = Swarm(300 + i,0,0,0)
        swarm.run_experiment(0.9)   
        from time import sleep
        sleep(1)
