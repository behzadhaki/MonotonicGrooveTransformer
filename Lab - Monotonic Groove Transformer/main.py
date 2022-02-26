# Import model
from utils import load_model, get_new_drum_osc_msgs
import torch


# Import OSC
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

if __name__ == '__main__':
    # ------------------ Load Trained Model  ------------------ #
    model_name = "groove_transformer_trained"
    model_path = f"trained_torch_models/{model_name}.model"

    groove_transformer = load_model(model_name, model_path)

    # ------  Create an empty an empty torch tensor
    input_tensor = torch.zeros((1, 32, 27))

    # ------  Create an empty h, v, o tuple for previously generated events to avoid duplicate messages
    (h_old, v_old, o_old) = (torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)))

    # get velocity and timing
    groove_velocities = torch.rand((32))
    groove_timings = -0.5 + torch.rand((32))

    """
    input_tensor[0, :, 2] = torch.where(groove_velocities > 0, 1, 0)  # set hits
    input_tensor[0, :, 11] = groove_velocities  # set velocities
    input_tensor[0, :, 20] = groove_timings  # set microtimings

    h, v, o = groove_transformer.predict(input_tensor)"""



    # ----------------------------------------------------------

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
        print(input_tensor[:, :, 9:18])

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

    while (1):
        server.handle_request()
        # get new generated pattern
        h_new, v_new, o_new = groove_transformer.predict(input_tensor, thres=0.1)

        # send to pd
        osc_messages_to_send = get_new_drum_osc_msgs((h_new, v_new, o_new))
        print("RESET TABLE!")
        py_to_pd_OscSender.send_message("/reset_table", 1)
        for (address, h_v_ix_tuple) in osc_messages_to_send:
            py_to_pd_OscSender.send_message(address, h_v_ix_tuple)

        # update old values with the sent ones
        # (h_old, v_old, o_old) = (h_new, v_new, o_new)

"""
if __name__ == '__main__':

    # Lists for storing received values
    velocity = [0]
    quitFlag = [False]

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
    def velocity_message_handler(address, *args):
        velocity[0] = args[0]

    # define the handler for quit message message
    def quit_message_handler(address, *args):
        quitFlag[0] = True
        print("QUITTING!")

    # pass the handlers to the dispatcher
    dispatcher.map("/global*", velocity_message_handler)
    dispatcher.map("/step*", quit_message_handler)

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

    while (quitFlag[0] is False):
        server.handle_request()
        print(f"Received velocity value {velocity[0]}")

        # 1. generate a random pitched note with the provided received value
        pitch = int(random.randrange(40, 52))       # 1 octave
        vel = velocity[0]
        duration = int(random.randrange(0, 1000))

        # 2. select a random drum voice
        drum_voice = drum_voices[random.randint(0, 5)]

        # 3. Send Notes to pd (send pitch last to ensure syncing)
        py_to_pd_OscSender.send_message("/gamelan/velocity_duration", (velocity, duration))
        py_to_pd_OscSender.send_message("/gamelan/pitch", pitch)

        py_to_pd_OscSender.send_message("/drum/velocity_duration", (velocity, duration))
        py_to_pd_OscSender.send_message("/drum/pitch", drum_voice_pitch_map[drum_voice])

    # ---------------------------------------------------------- #pythonosc
    
"""