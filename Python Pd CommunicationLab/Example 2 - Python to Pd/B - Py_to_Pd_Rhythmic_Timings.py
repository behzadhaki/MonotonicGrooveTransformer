import random
import time

#!pip install python-osc
from pythonosc.udp_client import SimpleUDPClient

def generate_rhythm_timings(total_duration_sec, rhythm_ioi_ratios):
    # total_duration_sec --> the duration of 1 loop
    # rhythm_ioi_ratios --> a list of inter-onset ratios, e.g. [1, 1, 1, 1] --> four notes with same ratio
    total_num_divisions = sum(rhythm_ioi_ratios)
    rhythm_timings = [float(ioi_ratio)*(float(total_duration_sec) / float(total_num_divisions))
                      for ioi_ratio in rhythm_ioi_ratios]
    return rhythm_timings

if __name__ == '__main__':

    # create an instance of the osc_sender class above
    ip='127.0.0.1'
    sending_to_port=1123
    py_to_pd_OscSender = SimpleUDPClient(ip, sending_to_port)

    # Send generated notes to pd
    number_of_note_to_gen = 1000

    # Specify and generate rhythm
    total_duration_sec = 4
    rhythm_ioi_ratios = [1, 3, 2, 2, 0.5, 1.5, 1, 1] # [1, 1, 1, 1]
    rhythm_timings = generate_rhythm_timings(total_duration_sec, rhythm_ioi_ratios)

    # Start generating each note with associated timing
    for ix in range(number_of_note_to_gen):
        # Generate pitch velocity and duration
        pitch = int(random.randrange(40, 70))
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
        print("\t\t\t Wait for {:.2f} seconds".format(rhythm_timings[ix % len(rhythm_timings)]))
        time.sleep(rhythm_timings[ix % len(rhythm_timings)])
