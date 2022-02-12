import random
import time

#!pip install python-osc
from pythonosc.udp_client import SimpleUDPClient


class OscSender:
    """
    Class for sending OSC messages from python to pd
    This class establishes a connection with pd server on ip address and sending_to_port
    """
    def __init__(self, ip, sending_to_port):
        """
        Constructor for OSC_SENDER CLASS

        :param ip: ip address of client ==> 127.0.0.1 (for local host/ inter app communication on same machine)
        :param sending_to_port: the port on which pure data listens for incoming data
        """
        self.ip = ip
        self.sending_to_port = sending_to_port

        self.client = SimpleUDPClient(self.ip, self.sending_to_port)

    def send_to_pd(self, message_parameters, message_values):
        """
        sends a list of messages to pd
        Note 1: Messages are sent in the same order as presented in the lists
        Note 2: ALWAYS USE LISTS EVEN IF SENDING A SINGLE PARAM/VALUE

        :param message_parameters: list of str: example ["/note/pitch", /note/duration"]
        :param message_values: list of ints, floats: example [53, 1000]
        """

        if len(message_parameters) != len(message_values):
            raise ValueError("The number of message_types do not match the values")

        else:
            for ix, param in enumerate(message_parameters):
                self.client.send_message(param, message_values[ix])

    def get_ip(self):
        return self.ip

    def get_sending_to_port(self):
        return self.sending_to_port

    def get_client(self):
        return self.client

    def change_ip_port(self, ip, port):
        self.ip = ip
        self.sending_to_port = port
        self.client = SimpleUDPClient(self.ip, self.sending_to_port)


class NoteGenerator:
    def __init__(self, min_pitch=48, pitch_semitone_range=24, vel_range=(0, 127), dur_range=(10, 1000)):
        self.pitch_range = (min_pitch, min_pitch+pitch_semitone_range)
        self.vel_range = vel_range
        self.dur_range = dur_range

    def generate(self, n_examples=1):

        message_parameters = ["/note/velocity", "/note/duration", "/note/pitch"]
        message_values = []

        for ix in range(n_examples):
            velocity = int(random.randrange(*self.vel_range))
            duration=int(random.randrange(*self.dur_range))
            pitch = int(random.randrange(*self.pitch_range))

            message_values.append([velocity, duration, pitch])

        return message_parameters, message_values


if __name__ == '__main__':

    # create an instance of the osc_sender class above
    py_to_pd = OscSender(ip='127.0.0.1', sending_to_port=1123)
    print("ip", py_to_pd.get_ip())

    # create an instance of the generator (generative model)
    note_generator = NoteGenerator(min_pitch=48, pitch_semitone_range=24,
                                   vel_range=(20, 90), dur_range=(110, 2000))

    # Generate the notes
    number_of_note_to_gen = 32
    message_parameters, message_values = note_generator.generate(n_examples=number_of_note_to_gen)

    # Let's send some random notes to pd

    for ix, vals in enumerate(message_values):

        py_to_pd.send_to_pd(message_parameters, vals)

        time.sleep(random.randrange(1, 10)/30)
