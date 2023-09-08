# coding: utf-8
import queue
import threading
import time
import requests
from local_task import TrainLoraTask
from log import get_logger
import traceback
mutex = threading.Lock()

next = True


def get_next():
    global next
    mutex.acquire()
    val = next
    mutex.release()
    return val


def set_next(val):
    global next
    mutex.acquire()
    next = val
    mutex.release()


class TaskFetcher(object):

    def __init__(self) -> None:
        pass

    def run(self, q: queue.Queue):
        pass


def fetch_one_task(done=None) -> TrainLoraTask:
    url = "https://test-zy.greatleapai.com//train/gettrain"
    headers = {
        "--ImFromYanCheng---": "x13413413jljkljalf13343jlkajdfkla",
        "Content-Type": "application/json"
    }
    rsp = requests.post(url, headers=headers, json={
        "name": "fechter_1"
    })

    rsp_data = rsp.json().get("data")

    task_id = rsp_data.get("task_id", "")
    if len(task_id) == 0 or task_id == "0":
        return None

    task_info = rsp_data.get("task_info", {})
    task_meta = rsp_data.get("task_meta", {})
    print(task_info)
    return TrainLoraTask(task_id, task_meta, task_info, done)


def fetch_train_task(q: queue.Queue):

    logger = get_logger()
    logger.info("Trying to fetching task!\n")
    import sys
    sys.stdout.flush()

    task_id = 0
    while True:
        while get_next() == False:
            logger.info(f"Wating Executing.....{task_id}\n")
            time.sleep(1)

        def done():
            global next

            set_next(True)
            logger.info(f"Task finished:{task_id}\n")

        try:
            task = fetch_one_task(done)
            if task == None:
                time.sleep(1)
                continue

            set_next(False)

            q.put(task)
            logger.info(f"Put to queue!!!{task.task_id}")
        except Exception as e:
            logger.info("Feching Task Failed:{}".format(
                traceback.format_exc()))
            continue

        logger.info("Sending Task To Queue:{task_id}")


class TrainTaskFecher(TaskFetcher):

    def __init__(self) -> None:
        super().__init__()

    def run(self, q: queue.Queue()):
        print("TrainTaskFecher")
        t = threading.Thread(target=fetch_train_task, args=(q,))
        t.daemon = True
        t.start()
        return t
