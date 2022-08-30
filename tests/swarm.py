from threading import Thread
import os
import json
import time
from worker import Worker

class Swarm:

    def __init__(self, workers_number, test_time, wait_time, image_path, debug=True):
        print(f"Init swarm #={workers_number}")
        self.workers_number = workers_number
        self.test_time = test_time
        self.wait_time = wait_time
        self.debug = debug
        self.image_path = image_path
    
    def run_experiment(self, interval=40, fps=20):
        print(f"Running experiment #={self.workers_number}")
        import time
        resp_time, count, success, tic =0, 0, 0, time.time()
        workers = []
        logset = set([])
        while True:
            old_workers = list(filter(lambda w: w.done, workers))
            workers = list(filter(lambda w: not w.done, workers))
            
            for worker in old_workers:
                resp_time += worker.log["response_time"]
                print(worker.log)
                count += 1
                success += 1 if worker.log["msg"] == "success" else 0
            int_sec = int(time.time() - tic) 
            if  int_sec % 1 == 0 and int_sec not in logset:
                logset.add(int_sec)
                print(success, count, "left", interval - (time.time() - tic))
            workers = workers 
            new_workers = [Worker(
                threadID=i,
                wait_time=self.wait_time,
                test_time=self.test_time,
                experiment=self.workers_number,
                image_path=self.image_path,
                debug=False
            ) for i in range(self.workers_number - len(workers))]
            for worker in new_workers:   
                time.sleep(1./fps)         
                worker.start()
            workers = workers + new_workers
            toc = time.time()
            if toc - tic > interval:
                print(toc -  tic, len(workers), len(old_workers))
                return resp_time, success, count
            
            
        
        
        



if __name__ == "__main__":
    successes, resp_times, counts = 0, 0,0 
    TOTAL_LEN= 1
    print("Started")
    for i in range(TOTAL_LEN):
        swarm = Swarm(300,0,0,0)
        resp_time, success, count = swarm.run_experiment(fps=60)
        successes += success
        counts += count
        resp_time += resp_time
        print(resp_time, success, count)
        print(f"Success rate= ({success}/{count}), response time = {resp_time/count}")
        
    successes = successes 
    resp_time = resp_time 
    print(f"Success rate= ({successes}/{counts}), response time = {resp_times/count}")
    time.sleep(1000)
    