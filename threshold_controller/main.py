import threading
import tomllib
import pylsl
import time

import numpy as np

from fire import Fire
from dareplane_utils.stream_watcher.lsl_stream_watcher import (
    StreamWatcher,
    pylsl_xmlelement_to_dict,
)

from dareplane_utils.general.time import sleep_s

from threshold_controller.utils.logging import logger


def init_lsl_outlet(cfg: dict) -> pylsl.StreamOutlet:
    n_channels = 1
    info = pylsl.StreamInfo(
        cfg["lsl_outlet"]["name"],
        cfg["lsl_outlet"]["type"],
        n_channels,
        cfg["lsl_outlet"]["nominal_freq_hz"],
        cfg["lsl_outlet"]["format"],
    )

    # enrich a channel name
    chns = info.desc().append_child("channels")
    ch = chns.append_child("channel")
    ch.append_child_value("label", "control_decoding")
    ch.append_child_value("unit", "AU")
    ch.append_child_value("type", "controller_output")
    ch.append_child_value("scaling_factor", "1")

    outlet = pylsl.StreamOutlet(info)

    return outlet


def connect_stream_watcher(config: dict) -> StreamWatcher:
    """Connect the stream watchers"""
    sw = StreamWatcher(
        config["stream_to_query"]["stream"],
        buffer_size_s=config["stream_to_query"]["buffer_size_s"],
    )
    sw.connect_to_stream()

    # if the outlet config is to be derived, calc from here
    if config["lsl_outlet"]["nominal_freq_hz"] == "derive":
        inlet_info = pylsl_xmlelement_to_dict(sw.inlet.info())
        config["lsl_outlet"]["nominal_freq_hz"] = float(
            inlet_info["info"]["nominal_srate"]
        )

    return sw


def compute_controller_output(inp: np.ndarray, th: float = 10_000) -> int:
    if inp[-1] > th:
        return 150
    else:
        return 10


def main(stop_event: threading.Event = threading.Event()):
    logger.setLevel(10)
    config = tomllib.load(
        open("./configs/threshold_controller_config.toml", "rb")
    )
    sw = connect_stream_watcher(config)
    outlet = init_lsl_outlet(config)

    tstart = pylsl.local_clock()
    th = config["controller"]["threshold"]
    sent = 0

    while not stop_event.is_set():
        sw.update()
        req_samples = int(
            config["lsl_outlet"]["nominal_freq_hz"]
            * (pylsl.local_clock() - tstart)
        ) - sent

        # This is only correct if the nominal_freq_hz is derived from the source stream
        if req_samples > 0 and sw.n_new > 0:
            ufbuffer = sw.unfold_buffer()
            cval = compute_controller_output(
                ufbuffer[-sw.n_new :].mean(axis=0), th=th
            )
            # logger.debug(
            #     f"Controller output: {cval}, {ufbuffer.shape=}, {sw.n_new=}, {ufbuffer[-5:]=}"
            # )
            # print(
            #     f"Controller output: {cval}, {len(ufbuffer[-sw.n_new :])}, {ufbuffer[-3:]}"
            # )

            # push only as many as are required -> rectified downsampling of
            # nominal_freq_hz < inlet sfreq
            # logger.debug(f"Pushing: {req_samples=}, {cval=}")
            for _ in range(req_samples):
                outlet.push_sample([cval])
            sent += req_samples
            sw.n_new = 0

        sleep_s(0.2 / config["lsl_outlet"]["nominal_freq_hz"])


def get_main_thread() -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    stop_event.clear()

    thread = threading.Thread(target=main, kwargs={"stop_event": stop_event})
    thread.start()

    return thread, stop_event


if __name__ == "__main__":
    Fire(main)
