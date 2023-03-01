# Import model
from py_utils.utils import *
import torch
import time

# Import OSC
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
import queue
import pickle
import os

# Parser for terminal commands
import argparse
parser = argparse.ArgumentParser(description='Monotonic Groove to Drum Generator')
parser.add_argument('--py2pd_port', type=int, default=1123,
                    help='Port for sending messages from python engine to pd (default = 1123)',
                    required=False)
parser.add_argument('--pd2py_port', type=int, default=1415,
                    help='Port for receiving messages sent from pd within the py program (default = 1415)',
                    required=False)
parser.add_argument('--wait', type=float, default=2,
                    help='minimum rate of wait time (in seconds) between two executive generation (default = 2 seconds)',
                    required=False)
parser.add_argument('--show_count', type=bool, default=True,
                    help='prints out the number of sequences generated',
                    required=False)
# parser.add_argument('--model', type=str, default="100",
#                     help='name of the model: (1) light_version: less computationally intensive, or '
#                          '(2) heavy_version: more computationally intensive',
#                     required=False)

args = parser.parse_args()

if __name__ == '__main__':
    # ------------------ Load Trained Model  ------------------ #
    model_path = f"trained_torch_models"
    model_name = "100.pth"
    show_count = args.show_count

    groove_transformer_vae = load_model(model_name, model_path)

    voice_thresholds = [0.01 for _ in range(9)]
    voice_max_count_allowed = [16 for _ in range(9)]

    # load per style means/stds of z encodings
    file = open(os.path.join(model_path, "z_means.pkl"), 'rb')
    z_means_dict = pickle.load(file)
    print(z_means_dict)
    file.close()
    file = open(os.path.join(model_path, "z_stds.pkl"), 'rb')
    z_stds_dict = pickle.load(file)

    print("Available styles: ", list(z_means_dict.keys()))
    # get_random_sample_from_style(style_="global",
    #                              model_=groove_transformer_vae,
    #                              voice_thresholds_=voice_thresholds,
    #                              voice_max_count_allowed_=voice_max_count_allowed,
    #                              z_means_dict=z_means_dict,
    #                              z_stds_dict=z_stds_dict,
    #                              scale_means_factor=1.0, scale_stds_factor=1.0)

    # ------  Create an empty an empty torch tensor
    input_tensor = torch.zeros((1, 32, 3))

    # ------  Create an empty h, v, o tuple for previously generated events to avoid duplicate messages
    (h_old, v_old, o_old) = (torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)))

    # set the minimum time needed between generations
    min_wait_time_btn_gens = args.wait


    # -----------------------------------------------------

    # ------------------ OSC ips / ports ------------------ #
    # connection parameters
    ip = "127.0.0.1"
    receiving_from_pd_port = args.pd2py_port
    sending_to_pd_port = args.py2pd_port
    message_queue = queue.Queue()
    # ----------------------------------------------------------

    # ------------------ OSC Receiver from Pd ------------------ #
    # create an instance of the osc_sender class above
    py_to_pd_OscSender = SimpleUDPClient(ip, sending_to_pd_port)
    # ---------------------------------------------------------- #

    def process_message_from_queue(address, args):
        if "VelutimeIndex" in address:
            input_tensor[:, int(args[2]), 0] = 1 if args[0] > 0 else 0  # set hit
            input_tensor[:, int(args[2]), 1] = args[0] / 127  # set velocity
            input_tensor[:, int(args[2]), 2] = args[1]  # set utiming
        elif "threshold" in address:
            voice_thresholds[int(address.split("/")[-1])] = 1-args[0]
        elif "max_count" in address:
            voice_max_count_allowed[int(address.split("/")[-1])] = int(args[0])
        elif "regenerate" in address:
            pass
        elif "time_between_generations" in address:
            global min_wait_time_btn_gens
            min_wait_time_btn_gens = args[0]
        else:
            print ("Unknown Message Received, address {}, value {}".format(address, args))
    # python-osc method for establishing the UDP communication with pd
    server = OscMessageReceiver(ip, receiving_from_pd_port, message_queue=message_queue)
    server.start()

    # ---------------------------------------------------------- #


    # ------------------ NOTE GENERATION  ------------------ #
    # drum_voice_pitch_map = {"kick": 36, 'snare': 38, 'tom-1': 47, 'tom-2': 42, 'chat': 64, 'ohat': 63}
    # drum_voices = list(drum_voice_pitch_map.keys())
    
    number_of_generations = 0
    count = 0
    while (1):
        address, args = message_queue.get()
        process_message_from_queue(address, args)

        # only generate new pattern when there isnt any other osc messages backed up for processing in the message_queue
        if (message_queue.qsize() == 0):

            # ----------------------------------------------------------------------------------------------- #
            # ----------------------------------------------------------------------------------------------- #
            # EITHER GENERATE USING GROOVE OR GENERATE A RANDOM PATTERN

            # case 1. generate using groove
            h_new, v_new, o_new = generate_from_groove(
                model_=groove_transformer_vae,
                in_groove=input_tensor,
                voice_thresholds_=voice_thresholds,
                voice_max_count_allowed_=voice_max_count_allowed)

            # case 2. generate randomly
            # h_new, v_new, o_new = generate_random_pattern(
            #     model_=groove_transformer_vae,
            #     voice_thresholds_=voice_thresholds,
            #     voice_max_count_allowed_=voice_max_count_allowed,
            #     means_dict=z_means_dict,
            #     stds_dict=z_stds_dict,
            #     style="funk"
            # )

            # ----------------------------------------------------------------------------------------------- #
            # ----------------------------------------------------------------------------------------------- #
            # send to pd
            osc_messages_to_send = get_new_drum_osc_msgs((h_new, v_new, o_new))
            number_of_generations += 1

            # First clear generations on pd by sending a message
            py_to_pd_OscSender.send_message("/reset_table", 1)

            # Then send over generated notes one at a time
            for (address, h_v_ix_tuple) in osc_messages_to_send:
                py_to_pd_OscSender.send_message(address, h_v_ix_tuple)

            if show_count:
                print("Generation #", count)

            # Message pd that sent is over by sending the counter value for number of generations
            # used so to take snapshots in pd
            py_to_pd_OscSender.send_message("/generation_count", count)

            count += 1

            time.sleep(min_wait_time_btn_gens)