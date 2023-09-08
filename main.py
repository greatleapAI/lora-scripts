# coding: utf-8
from scheduler import TaskRunner
from fechter import TrainTaskFecher
from log import get_logger

if __name__ == "__main__":
    fetchers = []
    get_logger().info("hello world!")

    fetchers.append(TrainTaskFecher())
    tr = TaskRunner(fetchers)
    tr.run()
