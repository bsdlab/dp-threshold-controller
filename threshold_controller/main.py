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
    """
    Initialize an LSL (Lab Streaming Layer) outlet for threshold controller output.

    This function creates an LSL outlet that streams threshold controller decisions
    as a single-channel stream.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing LSL outlet settings. Must include
        an "lsl_outlet" key with name, type, nominal_freq_hz, and format fields.

    Returns
    -------
    pylsl.StreamOutlet
        The LSL outlet for streaming the controller output.
    """
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
    """
    Connect to and configure the input stream watcher.

    This function initializes a StreamWatcher to monitor an input LSL stream.
    If the outlet frequency is set to "derive", the frequency is set equal to the
    nominal sampling rate of the input stream.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing stream connection settings.

    Returns
    -------
    StreamWatcher
        Connected StreamWatcher instance ready for data monitoring.
    """
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
    """
    Compute the output of the controller.

    This function applies a simple threshold-based decision rule to the input signal.
    If the most recent sample exceeds the threshold, it returns a high control value,
    otherwise a low control value. This is a placeholder for more complex
    decision-making logic that could be implemented in a real controller.

    Parameters
    ----------
    inp : np.ndarray
        Input signal array. The threshold comparison is applied to the last sample.
    th : float, default=10_000
        Threshold value for the decision boundary.

    Returns
    -------
    int
        Controller output: 150 if threshold exceeded, 10 otherwise.
    """
    if inp[-1] > th:
        return 150
    else:
        return 10


def main(stop_event: threading.Event = threading.Event()):
    """
    Main processing loop for the threshold controller.

    This function implements the main real-time processing loop for the threshold
    controller. It connects to an input LSL stream, applies threshold-based decisions
    to the incoming data, and streams the control outputs via an LSL outlet.
    The configuration is loaded from "./configs/threshold_controller_config.toml".

    Parameters
    ----------
    stop_event : threading.Event, optional
        Threading event to signal when processing should stop. Default creates
        a new Event object.
    """
    logger.setLevel(10)
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
            tlast = pylsl.local_clock()
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
            sw.n_new = 0

            sleep_s(0.9 / config["lsl_outlet"]["nominal_freq_hz"])


def get_main_thread() -> tuple[threading.Thread, threading.Event]:
    """
    Run the main processing loop in a separate thread.

    This function creates and starts a background thread that runs the main
    threshold controller loop. It allows the controller to be stopped via 
    the returned Event object.

    Returns
    -------
    tuple[threading.Thread, threading.Event]
        A tuple containing:
        - threading.Thread: The thread object running the controller loop
        - threading.Event: Event object that can be .set() to stop the controller
    """
    stop_event = threading.Event()
    stop_event.clear()

    thread = threading.Thread(target=main, kwargs={"stop_event": stop_event})
    thread.start()

    return thread, stop_event


if __name__ == "__main__":
    Fire(main)
