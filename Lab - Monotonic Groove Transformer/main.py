# Import model
from utils import load_model, get_new_drum_osc_msgs, get_prediction
import torch


# Import OSC
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

if __name__ == '__main__':
    # ------------------ Load Trained Model  ------------------ #
    model_name = "groove_transformer_trained_2"         # "groove_transformer_trained"
    model_path = f"trained_torch_models/{model_name}.model"

    groove_transformer = load_model(model_name, model_path)

    voice_thresholds = [0.01 for _ in range(9)]
    voice_max_count_allowed = [16 for _ in range(9)]

    # ------  Create an empty an empty torch tensor
    input_tensor = torch.zeros((1, 32, 27))

    # ------  Create an empty h, v, o tuple for previously generated events to avoid duplicate messages
    (h_old, v_old, o_old) = (torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)))

    # get velocity and timing
    groove_velocities = torch.rand((32))
    groove_timings = -0.5 + torch.rand((32))
    # -----------------------------------------------------

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
        input_tensor[:, int(args[2]), 2] = 1 if args[0] > 0 else 0               # set hit
        input_tensor[:, int(args[2]), 11] = args[0]  / 127      # set velocity
        input_tensor[:, int(args[2]), 20] = args[1]          # set utiming

    # define the handler for sampling thresholds
    def sampling_threshold_handler(address, *args):
        voice_thresholds[int(address.split("/")[-1])] = 1-args[0]

    # define the handler for sampling thresholds
    def max_count_handler(address, *args):
        voice_max_count_allowed[int(address.split("/")[-1])] = int(args[0])

    # pass the handlers to the dispatcher
    dispatcher.map("/VelutimeIndex*", groove_event_handler)
    dispatcher.map("/threshold*", sampling_threshold_handler)
    dispatcher.map("/max_count*", max_count_handler)

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

    while (1):
        server.handle_request()
        # get new generated pattern
        # h_new, v_new, o_new = groove_transformer.predict(input_tensor, thres=0.5)
        h_new, v_new, o_new = get_prediction(groove_transformer, input_tensor, voice_thresholds, voice_max_count_allowed)
        _h, v, o = groove_transformer.forward(input_tensor)

        # send to pd
        osc_messages_to_send = get_new_drum_osc_msgs((h_new, v_new, o_new))
        print("RESET TABLE!")
        py_to_pd_OscSender.send_message("/reset_table", 1)
        for (address, h_v_ix_tuple) in osc_messages_to_send:
            py_to_pd_OscSender.send_message(address, h_v_ix_tuple)