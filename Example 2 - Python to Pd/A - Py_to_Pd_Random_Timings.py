import random
import time

#!pip install python-osc
from pythonosc.udp_client import SimpleUDPClient

if __name__ == '__main__':

    # create an instance of the osc_sender class above
    ip='127.0.0.1'
    sending_to_port=1123
    py_to_pd_OscSender = SimpleUDPClient(ip, sending_to_port)

    # Send generated notes to pd
    number_of_note_to_gen = 1000

    for ix in range(number_of_note_to_gen):
        # Generate pitch velocity and duration
        pitch = int(random.randrange(40, 64))
        velocity = int(random.randrange(10, 127))
        duration = int(random.randrange(0, 1000))

        # output playback information
        print(f"\n Playing Note {ix}")

        #Send the note to pd
        message_parameters = ["/note/pitch", "/note/velocity", "/note/duration"]
        print(f"Sending OSC | /note/pitch/{pitch} | /note/velocity/{velocity} | /note/duration/{duration}")

        py_to_pd_OscSender.send_message("/note/velocity", velocity)
        py_to_pd_OscSender.send_message("/note/duration", duration)
        py_to_pd_OscSender.send_message("/note/pitch", pitch)

        # wait for a random period of 0.5 to 2 second before playing back the next note
        wait_time_before_next_note = random.randrange(1, 10)/30    # this is basically IOI (inter-onset interval)
        print("\t\t\t Wait for {:.2f} seconds".format(wait_time_before_next_note))
        time.sleep(wait_time_before_next_note)
