import threading
import tomllib
import pylsl

import numpy as np

from fire import Fire
from dareplane_utils.stream_watcher.lsl_stream_watcher import (
    StreamWatcher,
    pylsl_xmlelement_to_dict,
)


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
    config = tomllib.load(
        open("./configs/threshold_controller_config.toml", "rb")
    )
    sw = connect_stream_watcher(config)
    outlet = init_lsl_outlet(config)

    tlast = pylsl.local_clock()
    th = config["controller"]["threshold"]

    while not stop_event.is_set():
        sw.update()
        req_samples = int(
            config["lsl_outlet"]["nominal_freq_hz"]
            * (pylsl.local_clock() - tlast)
        )

        # This is only correct if the nominal_freq_hz is derived from the source stream
        if req_samples > 0 and sw.n_new > 0:
            ufbuffer = sw.unfold_buffer()
            cval = compute_controller_output(ufbuffer, th=th)
            # logger.debug(f"Controller output: {cval}")
            # print(
            #     f"Controller output: {cval}, {len(ufbuffer[-sw.n_new :])}, {ufbuffer[-3:]}"
            # )
            for _ in ufbuffer[-sw.n_new :]:
                outlet.push_sample([cval])
            sw.n_new = 0
            tlast = pylsl.local_clock()


def get_main_thread() -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    stop_event.clear()

    thread = threading.Thread(target=main, kwargs={"stop_event": stop_event})
    thread.start()

    return thread, stop_event


if __name__ == "__main__":
    Fire(main)
