# Import model
import torch
from helpers.VAE.modelLoader import load_variational_mgt_model
import os
import numpy as np
import time

import threading

#!pip install python-osc
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

# MAGENTA MAPPING
ROLAND_REDUCED_MAPPING = {
    "KICK": [36],
    "SNARE": [38, 37, 40],
    "HH_CLOSED": [42, 22, 44],
    "HH_OPEN": [46, 26],
    "TOM_3_LO": [43, 58],
    "TOM_2_MID": [47, 45],
    "TOM_1_HI": [50, 48],
    "CRASH": [49, 52, 55, 57],
    "RIDE":  [51, 53, 59]
}
ROLAND_REDUCED_MAPPING_VOICES = ['/KICK', '/SNARE', '/HH_CLOSED', '/HH_OPEN', '/TOM_3_LO', '/TOM_2_MID', '/TOM_1_HI', '/CRASH', '/RIDE']


def encode_into_latent_z(model_, in_groove):
    """ returns Z corresponding to a provided groove of shape (1, 32, 3)
    """
    # Generation using an Input Groove
    # Step 1. Get the mean and var of the latent encoding
    mu, logvar = model_.encode_to_mu_logvar(in_groove)

    # Step 2. Sample a latent vector (z) from latent distribution
    latent_z = model_.reparametrize(mu, logvar)

    return latent_z


def generate_from_groove(model_, in_groove, voice_thresholds_, voice_max_count_allowed_):
    """ returns a full drum pattern based on the provided input groove
    """
    latent_z = encode_into_latent_z(model_, in_groove)

    # Step 3. Generate using the latent encoding
    return model_.sample(
        latent_z=latent_z,
        voice_thresholds=voice_thresholds_,
        voice_max_count_allowed=voice_max_count_allowed_,
        return_concatenated=False,
        sampling_mode=0)

def generate_random_pattern(model_, voice_thresholds_, voice_max_count_allowed_,
                            means_dict, stds_dict, style):
    assert style in means_dict, f"Style must be one of {list(means_dict.keys())}"
    means = means_dict[style]
    stds = stds_dict[style]
    random_z = [np.random.normal(loc=means[ix], scale=stds[ix]) for ix in range(128)]
    # Step 3. Generate using the latent encoding
    return model_.sample(
        latent_z=torch.tensor(random_z, dtype=torch.float32),
        voice_thresholds=voice_thresholds_,
        voice_max_count_allowed=voice_max_count_allowed_,
        return_concatenated=False,
        sampling_mode=0)

def decode_z_into_drums(model_, latent_z, voice_thresholds, voice_max_count_allowed):
    """ returns a full drum pattern based on a provided latent encoding Z
    """
    return model_.sample(latent_z=torch.tensor(latent_z, dtype=torch.float32),
                         voice_thresholds=voice_thresholds,
                         voice_max_count_allowed=voice_max_count_allowed,
                         return_concatenated=False,
                         sampling_mode=0)

def get_random_sample_from_style(style_, model_,
                                 voice_thresholds_,
                                 voice_max_count_allowed_,
                                 z_means_dict, z_stds_dict,
                                 scale_means_factor=1.0, scale_stds_factor=1.0):

    assert style_ in z_means_dict.keys(), f"Make sure the style is one of {list(z_means_dict.keys())}"
    z = np.random.normal(loc=np.array(z_means_dict[style_]) * scale_means_factor,
                         scale=np.array(z_stds_dict[style_]) * scale_stds_factor)
    h_,v_,o_ = decode_z_into_drums(model_, z, voice_thresholds_, voice_max_count_allowed_)
    return z, (h_, v_, o_)

def load_model(model_name, model_path):

    # Initialize model
    groove_transformer = load_variational_mgt_model(os.path.join(model_path, model_name))
    

    # put in evaluation mode
    groove_transformer.eval()

    return groove_transformer

def get_new_drum_osc_msgs(hvo_tuple_new, hvo_tuple_prev=None):
    message_list = list()
    (h_new, v_new, o_new) = hvo_tuple_new
    hit_locations_new = torch.nonzero(h_new)

    for batch_ix, timestep, voiceIx  in hit_locations_new:
        # timestep = int(timestep)
        # voiceIx = int(voiceIx)
        if hvo_tuple_prev is not None:
            if (hvo_tuple_prev[0][batch_ix, timestep, voiceIx] != h_new[batch_ix, timestep, voiceIx] or
                    hvo_tuple_prev[1][batch_ix, timestep, voiceIx] != v_new[batch_ix, timestep, voiceIx] or
                    hvo_tuple_prev[2][batch_ix, timestep, voiceIx] != o_new[batch_ix, timestep, voiceIx]):
                address = ROLAND_REDUCED_MAPPING_VOICES[voiceIx]
                osc_msg = (
                        "/drum_generated".join(address),
                        (v_new[batch_ix, timestep, voiceIx].item(), o_new[batch_ix, timestep, voiceIx].item(),
                        timestep.item())
                    )
                message_list.append(osc_msg)
        else:
            address = ROLAND_REDUCED_MAPPING_VOICES[voiceIx]
            osc_msg = (
                "/drum_generated"+address,
                (v_new[batch_ix, timestep, voiceIx].item(), o_new[batch_ix, timestep, voiceIx].item(),
                 timestep.item())
            )
            message_list.append(osc_msg)

            # print("OSC msg to send: ", osc_msg)


    return message_list


def get_prediction(trained_model, input_tensor, voice_thresholds, voice_max_count_allowed, sampling_mode=0):
    trained_model.eval()
    return generate_from_groove(trained_model, input_tensor, voice_thresholds, voice_max_count_allowed)


class OscMessageReceiver(threading.Thread):

    def __init__(self, ip, receive_from_port, message_queue):
        """
        Constructor for OscReceiver CLASS

        :param ip:                      ip address that pd uses to send messages to python
        :param receive_from_port:       port  that pd uses to send messages to python
        :param quit_event:              a Threading.Event object, for finishing the receiving process
        :param address_list:            list of osc addresses that need to be assigned a specific handler
        :param address_handler_list:    the handlers for a received osc message
        """

        # we want the OscReceiver to run in a separate concurrent thread
        # hence it is a child instance of the threading.Thread class
        super(OscMessageReceiver, self).__init__()

        # connection parameters
        self.ip = ip
        self.receiving_from_port = receive_from_port
        self.message_queue = message_queue

        # dispatcher is used to assign a callback to a received osc message
        self.dispatcher = Dispatcher()
        self.dispatcher.set_default_handler(self.default_handler)

        # python-osc method for establishing the UDP communication with pd
        self.server = BlockingOSCUDPServer((self.ip, self.receiving_from_port), self.dispatcher)

    def run(self):
        # When you start() an instance of the class, this method starts running
        print("OscMessageReceiver Started ---")

        # Counter for the number messages are received
        count = 0

        # Keep waiting of osc messages (unless the you've quit the receiver)
        while 1:
            # handle_request() waits until a new message is received
            # Messages are buffered! so if each loop takes a long time, messages will be stacked in the buffer
            # uncomment the sleep(1) line to see the impact of processing time
            self.server.handle_request()
            count = (count + 1)  # Increase counter
            # time.sleep(1)

    def default_handler(self, address, *args):
        self.message_queue.put((address, args))


    def get_ip(self):
        return self.ip

    def get_receiving_from_port(self):
        return self.receiving_from_port

    def get_server(self):
        return self.server

    def change_ip_port(self, ip, port):
        self.ip = ip
        self.receiving_from_port = port
        self.server = BlockingOSCUDPServer(self.ip, self.receiving_from_port)