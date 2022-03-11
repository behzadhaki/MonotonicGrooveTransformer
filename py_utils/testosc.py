# Import model
from py_utils.utils import load_model, get_new_drum_osc_msgs
import torch


# Import OSC
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

if __name__ == '__main__':

    # ------------------ OSC ips / ports ------------------ #
    # connection parameters
    ip = "127.0.0.1"
    receiving_from_pd_port = 1415
    sending_to_pd_port = 1123

    # ----------------------------------------------------------

    # ------------------ OSC Receiver from Pd ------------------ #
    # create an instance of the osc_sender class above
    py_to_pd_OscSender = SimpleUDPClient(ip, sending_to_pd_port)
    # ---------------------------------------------------------- #

    # ------------------ OSC Receiver from Pd ------------------ #
    # dispatcher is used to assign a callback to a received osc message
    # in other words the dispatcher routes the osc message to the right action using the address provided
    dispatcher = Dispatcher()


    # define the handler for messages starting with /velocity]
    def groove_event_handler(address, *args):
        pass

    # pass the handlers to the dispatcher
    dispatcher.map("/VelutimeIndex*", groove_event_handler)


    # you can have a default_handler for messages that don't have dedicated handlers
    def default_handler(address, *args):
        print(f"No action taken for message {address}: {args}")

    dispatcher.set_default_handler(default_handler)

    # python-osc method for establishing the UDP communication with pd
    server = BlockingOSCUDPServer((ip, receiving_from_pd_port), dispatcher)
    # ---------------------------------------------------------- #

    # ------------------ NOTE GENERATION  ------------------ #
    drum_voice_pitch_map = {"kick": 36, 'snare': 38, 'tom-1': 47, 'tom-2': 42, 'chat': 64, 'ohat': 63}
    drum_voices = list(drum_voice_pitch_map.keys())
    py_to_pd_OscSender.send_message("/reset_table", 1)
    v_o_ix_tuple = (1, -.1, 2)
    py_to_pd_OscSender.send_message("/drum_generated/KICK", v_o_ix_tuple)
    v_o_ix_tuple = (.3, -.1, 20)
    py_to_pd_OscSender.send_message("/drum_generated/HH_OPEN", v_o_ix_tuple)




