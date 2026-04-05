from collections import defaultdict, deque
import datetime
import json
import logging
import time

import torch

import dinov2.distributed as distributed

logger = logging.getLogger("dinov2")


class MetricLogger(object):
    def __init__(self, delimiter="    ", output_file=None, window_size=20):
        self.meters = defaultdict(lambda: Metric(window_size))
        self.delimiter = delimiter
        self.output_file = output_file
        self.dataloader = None
        self.log_msg = None

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.avg()))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.avg() for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def build_log_msg(self, header, n_iterations):
        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{i" + space_fmt + "}/{total}]",
            "eta: {eta}",
            "{meters}",
            "time: {time:.4f}",
            "data: {data:.4f}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        return self.delimiter.join(log_list)

    def log_iteration(self, i, n_iterations, eta, iter_time, data_time):

        MB = 1024.0 * 1024.0
        meters = str(self)
        msg_values = {
            "i": i,
            "total": n_iterations,
            "eta": eta,
            "meters": meters,
            "time": iter_time,
            "data": data_time,
        }
        if torch.cuda.is_available():
            msg_values["memory"] = torch.cuda.max_memory_allocated() / MB

        logger.info(self.log_msg.format(**msg_values))  # type: ignore

    def log_every(
        self,
        print_freq,
        header=None,
        n_iterations=None,
        start_iteration=0,
        grad_accum_steps=1,
    ):
        iterable = self.dataloader
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        data_end = time.time()
        iter_end = time.time()
        iter_time = Metric()
        data_time = Metric()

        if n_iterations is None:
            n_iterations = len(iterable)  # type: ignore
        self.log_msg = self.build_log_msg(header, n_iterations)

        grad_accum_counter = 0
        for obj in iterable:  # type: ignore
            data_time.update(time.time() - data_end)
            yield obj
            if (grad_accum_counter + 1) % grad_accum_steps == 0:
                iter_time.update(time.time() - iter_end)
                if i % print_freq == 0 or i == n_iterations - 1:
                    self.dump_in_output_file(i, iter_time.avg(), data_time.avg())
                    eta_seconds = iter_time.avg() * (n_iterations - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    self.log_iteration(
                        i, n_iterations, eta_string, iter_time.avg(), data_time.avg()
                    )
                i += 1
                iter_end = time.time()
            grad_accum_counter += 1
            data_end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / n_iterations
            )
        )


class Metric:
    """Track a series of values and provide the recent average."""

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)

    def update(self, value):
        self.deque.append(value)

    def avg(self):
        return sum(self.deque) / len(self.deque) if self.deque else 0.0
