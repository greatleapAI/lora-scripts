# coding:utf-8

from .config import ZY_BACK_URL, TRAINER_NAME
from .log import get_logger
import enum
import requests


class LoraClient(object):
    FecthUrl = f"{ZY_BACK_URL}/train/gettrain"
    UpdateStatusUrl = f"{ZY_BACK_URL}/train/updatestatus"
    UpdateProgressUrl = f"{ZY_BACK_URL}/train/updateprogress"
    FinshTaskUrl = f"{ZY_BACK_URL}/train/finishtask"

    def __init__(self, name=TRAINER_NAME) -> None:
        self.headers = {
            "--ImFromYanCheng---": "x13413413jljkljalf13343jlkajdfkla",
            "Content-Type": "application/json"
        }
        self.name = name

    def call_zy_backend(self, url, data):

        data["trainer"] = self.name
        rsp = requests.post(url, headers=self.headers, json=data)

        if rsp.status_code != 200:
            get_logger().error(f"callback end failed: context={rsp.text}")
            return False, None

        rsp_data: dict = rsp.json()
        code = rsp_data.get("status_code")
        msg = rsp_data.get("status_msg")

        if code != 200:
            get_logger().error(
                f"Call Failed:url={url}, code={code}, msg={msg}")
            return False, {}

        return True, rsp_data.get("data", {})

    def fetch(self):
        return self.call_zy_backend(self.FecthUrl, {})

    def update_status(self, task_id: int, status: enum.Enum):
        succ, rsp = self.call_zy_backend(self.UpdateStatusUrl, {
            "task_id": str(task_id),
            "status": status.value
        })

        return succ

    def update_progress(self, task_id, progress):
        return self.call_zy_backend(self.UpdateProgressUrl, {
            "task_id": str(task_id),
            "progress": progress
        })

    def finish_task(self,  succ: int, task_id, result: dict):
        return self.call_zy_backend(self.FinshTaskUrl, {
            "succ": succ,
            "result": result,
            "task_id": str(task_id),
        })


__lora_client = LoraClient()


def get_lora_client():
    return __lora_client
