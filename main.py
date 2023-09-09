# coding: utf-8
from zytask.log import get_logger
from zytask.fechter import TrainTaskFecher
from zytask.scheduler import TaskRunner


if __name__ == "__main__":
    fetchers = []
    get_logger().info("hello world!")

    fetchers.append(TrainTaskFecher())
    tr = TaskRunner(fetchers)
    tr.run()
