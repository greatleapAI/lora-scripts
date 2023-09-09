import queue
import threading
from .local_task import Task
from .log import get_logger
import traceback
from .fechter import TaskFetcher
from .urls_manager import get_lora_client
import enum

TaskStatus = enum.Enum('TaskStatus', ('New', 'Tagging', 'Waiting', 'Fetched',
                                      'Prepareing', 'Training', 'Uploading', 'Abort', 'Succ'))


def worker(q: queue.Queue):
    logger = get_logger()
    logger.info("working!!!!")

    client = get_lora_client()

    while True:
        task: Task = q.get()
        logger.info("Handling:{}".format(task.task_id))

        try:

            task_id = task.task_id

            if not client.update_status(task_id, TaskStatus.Prepareing):
                continue
            if not task.prepare():
                logger.error(f"prepare task failed: {task.task_id}")
                client.finish_task(0, task_id, task.get_result())
                continue

            if not client.update_status(task_id, TaskStatus.Training):
                continue
            if not task.run():
                logger.error(f"run task failed:{task.task_id}")
                client.finish_task(0, task_id, task.get_result())
                continue

            if not client.update_status(task_id, TaskStatus.Uploading):
                continue

            if not task.push_result():
                logger.error(f"push result failed: {task.task_id}")
                client.finish_task(0, task_id, task.get_result())
                continue

            client.finish_task(1, task_id, task.get_result())

        except Exception as e:
            logger.error(traceback.format_exc())
            client.finish_task(
                0, task_id, {"exception": traceback.format_exc()})
        finally:
            task.finish()

        logger.info("task handling finished{}".format(task.task_id))


class TaskRunner(object):

    def __init__(self, fetcher: list[TaskFetcher]) -> None:
        self.task_q = queue.Queue()
        self.fetcher = fetcher
        self.threads = []

        for f in self.fetcher:
            print("ffff")
            t = f.run(self.task_q)
            self.threads.append(t)

    def run(self):
        t = threading.Thread(target=worker, args=(self.task_q,))
        t.daemon = True
        t.start()
        t.join()
