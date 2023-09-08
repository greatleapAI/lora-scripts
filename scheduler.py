import queue
import threading
from local_task import Task
import log
import traceback
from fechter import TaskFetcher


def worker(q: queue.Queue):
    logger = log.get_logger()
    logger.info("working!!!!")
    while True:
        task: Task = q.get()
        logger.info("Handling:{}".format(task.task_id))
        try:
            if not task.prepare():
                logger.error(f"prepare task failed: {task.task_id}")
                task.fail_report()
                continue
            if not task.run():
                logger.error(f"run task failed:{task.task_id}")
                continue

            task.success_report()

        except Exception as e:
            logger.error(traceback.format_exc())
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
