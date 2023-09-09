# coding: utf-8
from .config import TRAIN_TMP_OUT_ROOT, TRAIN_TMP_PATH_ROOT
import oss2
import json
from .common import mkdir_p
from .log import get_logger
import os


default_args = {
    "save_model_as": "safetensors",
    "network_module": "networks.lora",
    "network_weights": "",
    "network_dim": 32,
    "network_alpha": 32,
    "resolution": "512,512",   # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数
    "batch_size": 1,           # batch size
    "max_train_epoches": 10,   # max train epoches | 最大训练 epoch
    "save_every_n_epochs": 2,  # save every n epochs | 每 N 个 epoch 保存一次
    "train_unet_only": 0,          # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
    "train_text_encoder_only": 0,  # train Text Encoder only | 仅训练 文本编码器
    "stop_text_encoder_training": 0,  # stop text encoder training | 在第N步时停止训练文本编码器
    "noise_offset": 0,   # noise offset | 在训练中添加噪声偏移来改良生成非常暗或者非常亮的图像，如果启用，推荐参数为0.1
    # keep heading N tokens when shuffling caption tokens | 在随机打乱 tokens 时，保留前 N 个不变。
    "keep_tokens": 0,
    # minimum signal-to-noise ratio (SNR) value for gamma-ray | 伽马射线事件的最小信噪比（SNR）值  默认为 0
    "min_snr_gamma": 0,
    "lr": "1e-4",
    "unet_lr": "1e-4",
    "text_encoder_lr": "1e-5",
    # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"
    "lr_scheduler": "cosine_with_restarts",
    # warmup steps | 学习率预热步数，lr_scheduler 为 constant 或 adafactor 时该值需要设为0。
    "lr_warmup_steps": 0,
    # cosine_with_restarts restart cycles | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时起效。
    "lr_restart_cycles": 1,
    "min_bucket_reso": 256,              # arb min resolution | arb 最小分辨率
    "max_bucket_reso": 1024,             # arb max resolution | arb 最大分辨率
    # persistent dataloader workers | 容易爆内存，保留加载训练集的worker，减少每个 epoch 之间的停顿
    "persistent_data_loader_workers": 0,
    "clip_skip": 2,                       # clip skip | 玄学 一般用 2
    # Optimizer type | 优化器类型 默认为 AdamW8bit，可选：AdamW AdamW8bit Lion SGDNesterov SGDNesterov8bit DAdaptation AdaFactor
    "optimizer_type": "AdamW",
    # LyCORIS network algo | LyCORIS 网络算法 可选 lora、loha、lokr、ia3、dylora。lora即为locon
    "algo": "lora",
    "conv_dim": 4,     # conv dim | 类似于 network_dim，推荐为 4
    "conv_alpha": 4,  # conv alpha | 类似于 network_alpha，可以采用与 conv_dim 一致或者更小的值
    # dropout | dropout 概率, 0 为不使用 dropout, 越大则 dropout 越多，推荐 0~0.5， LoHa/LoKr/(IA)^3暂时不支持
    "dropout": "0",
    "output_name": "test"

}

auth = oss2.Auth("LTAI5tQnLUQgZSY9xy7rz2fL",
                 "eIVPKw6R7eSv2Mt2EJ6ZJJ9Rh3HKqJ")
bucket: oss2.Bucket = oss2.Bucket(
    auth, 'oss-cn-hangzhou.aliyuncs.com/', 'zy-pic-items-test')


model_store_path = "trained_lora"


class Task(object):

    def __init__(self, task_id, done=None) -> None:
        self.task_id = task_id
        self.done = done
        self.result = {}

    def prepare(self) -> bool:
        return True

    def run(self) -> bool:
        pass

    def push_result(self) -> bool:
        pass

    def finish(self):
        if self.done == None:
            return

        if callable(self.done):
            self.done()

    def __str__(self) -> str:
        return "task_id{}".format(self.task_id)

    def get_result(self):
        return self.result


class TrainLoraTask(Task):
    def __init__(self, task_id, task_meta, task_info, done=None) -> None:
        self.task_id = task_id
        self.task_info = task_info
        self.params = task_info.get("params", '{}')
        self.task_meta = task_meta
        self.local_sh = "python -m accelerate.commands.launch"
        self.train_data_dir = ""
        self.train_data_output = ""
        self.train_logs_dir = ""
        self.train_result_path = ""
        self.output_name = ""

        super().__init__(task_id, done)

    def append_args(self, args):
        self.local_sh += " " + args + " "

    def append_by_key(self, ak):
        return self.append_kv(ak, ak)

    def append_kv(self, ak, k):
        av = self.get_param(k, None)

        if av == None:
            get_logger().info(f"args not found:{k}, ignore")
            return
        self.append_raw_kv(ak, av)

    def append_raw_kv(self, k, v):
        if type(v) == str:
            self.append_args('''--{}="{}"'''.format(k, v))
        else:
            self.append_args("--{}={}".format(k, v))

    def append_launch(self):
        pass

    def get_param(self, k, default):
        ret = self.params.get(k, default)
        if ret == default:
            return default_args.get(k, default)
        return ret

    def append_extra(self):

        model_info = self.params.get("model_info", {})
        v2 = model_info.get("v2", None)
        if v2 == None:
            self.append_by_key("clip_skip")
        else:
            self.append_args("--v2")

        train_unet_only = self.get_param("train_unet_only", 0)
        if train_unet_only == 1:
            self.append_args("--network_train_unet_only")

        train_text_encoder_only = self.get_param(
            "train_text_encoder_only", 0)

        if train_text_encoder_only == 1:
            self.append_args("--network_train_text_encoder_only")

        network_weights = self.get_param("network_weights", "")
        if network_weights != "":
            self.append_args("--network_weights {}".format(network_weights))

        optimizer_type = self.get_param("optimizer_type", "")
        if optimizer_type != "":
            self.append_args("--optimizer_type {}".format(optimizer_type))

        if optimizer_type == "DAdaptation":
            self.append_args("--optimizer_args decouple=True")

        network_module = self.get_param("network_module", None)
        if network_module == "lycoris.kohya":
            self.append_args("--network_args conv_dim={} conv_alpha={} algo={} dropout={}".format(
                self.get_param("conv_dim", ""), self.get_param(
                    "conv_alpha", ""), self.get_param("algo", ""), self.get_param("dropout", "")
            ))

        stop_text_encoder_training = self.get_param(
            "stop_text_encoder_training", 0)
        if stop_text_encoder_training != 0:
            self.append_args(
                "--stop_text_encoder_training {}".format(stop_text_encoder_training))

        noise_offset = self.get_param("noise_offset", "0")
        if noise_offset != "0":
            self.append_args('''--noise_offset "{}"'''.format(noise_offset))

        min_snr_gamma = self.get_param("min_snr_gamma", 0)
        if min_snr_gamma != 0:
            self.append_args("--min_snr_gamma {}".format(min_snr_gamma))

        self.append_args("--log_with=tensorboard")

    def prepare_shell_args(self):

        #
        model_info = self.params.get("model_info", {})
        self.params["model_path"] = "/data/shared/models/" + \
            model_info.get("path", "")

        self.append_launch()
        self.append_args("--num_cpu_threads_per_process=8")
        self.append_args("--num_processes=1")
        self.append_args('''\"./sd-scripts/train_network.py\"''')
        self.append_args("--enable_bucket")
        self.append_raw_kv("train_data_dir", self.train_data_dir)
        self.append_raw_kv("output_dir", self.train_data_output)
        self.append_raw_kv("logging_dir", self.train_logs_dir)

        arg_mps = []
        arg_mps.append(["log_prefix", "output_name"])
        arg_mps.append(["pretrained_model_name_or_path", "model_path"])
        arg_mps.append("resolution")
        arg_mps.append("network_module")
        arg_mps.append(["max_train_epochs", "max_train_epoches"])
        arg_mps.append(["learning_rate", "lr"])

        arg_mps.append("unet_lr")
        arg_mps.append("text_encoder_lr")
        arg_mps.append("lr_scheduler")
        arg_mps.append("lr_warmup_steps")
        arg_mps.append(["lr_scheduler_num_cycles", "lr_restart_cycles"])
        arg_mps.append("network_dim")
        arg_mps.append("network_alpha")
        arg_mps.append("output_name")
        arg_mps.append(["train_batch_size", "batch_size"])
        arg_mps.append("save_every_n_epochs")

        for arg in arg_mps:
            if type(arg) == str:
                self.append_kv(arg, arg)
            elif type(arg) == list:
                self.append_kv(arg[0], arg[1])

        self.append_raw_kv("mixed_precision", '''"fp16"''')
        self.append_raw_kv("save_precision", '''"fp16"''')
        self.append_raw_kv("seed", '''"1377"''')
        self.append_args("--cache_latents")
        self.append_raw_kv("prior_loss_weight", "1")
        self.append_raw_kv("max_token_length", "225")
        self.append_raw_kv("caption_extension", '''".txt"''')
        self.append_kv("save_model_as", "save_model_as")
        self.append_kv("min_bucket_reso", "min_bucket_reso")
        self.append_kv("max_bucket_reso", "max_bucket_reso")
        self.append_kv("keep_tokens", "keep_tokens")
        self.append_args("--xformers")
        self.append_args("--shuffle_caption")

        self.append_extra()

    def prepare(self) -> bool:
        logger = get_logger()
        to_path = TRAIN_TMP_PATH_ROOT + "/" + self.task_id
        self.train_data_dir = to_path
        mkdir_p(to_path)

        to_path = to_path + "/1"
        out_path = TRAIN_TMP_OUT_ROOT + "/" + self.task_id

        mkdir_p(out_path)
        mkdir_p(to_path)
        self.output_name = self.get_param(
            "output_name", "") + "." + self.get_param("save_model_as", "")

        self.train_result_path = out_path + "." + \
            self.get_param("save_model_as", "")

        self.train_data_output = out_path
        self.train_logs_dir = out_path

        oss_root = self.task_info.get("data_root", "")

        files: map = self.task_info["files"]

        for file_id, file_info in files.items():
            local_path = to_path + "/" + file_info["file_name"]
            local_text_path = to_path + "/" + file_id + ".txt"

            file_path = oss_root + "/" + file_info["file_name"]

            bucket.get_object_to_file(
                file_path, local_path)

            logger.info(f"Downloaded {file_path}->{local_path}")

            with open(local_text_path, "w") as f:
                f.write(file_info["tags_en"])
                f.write("\n")
                print(f"Write Tags{local_text_path}")

        self.prepare_shell_args()

        return super().prepare()

    def run(self) -> bool:
        get_logger().info(self.local_sh)
        ret = os.popen(self.local_sh)
        if not os.path.isfile(self.train_result_path):
            self.result = {
                "exec_stdout": "No File Generated, check train logs",
            }
            return False

        return True

    def push_result(self) -> bool:
        oss_path = model_store_path + "/" + self.task_id + "/" + self.output_name

        res = bucket.put_object_from_file(
            oss_path, self.train_result_path)
        if res.status != 200:
            self.result = {
                "local": self.train_result_path,
                "push_code": res.status,
                "push_res": res.resp.text,
            }
            return False

        self.result = {
            "local": self.train_result_path,
            "oss": oss_path,
        }

        return True
