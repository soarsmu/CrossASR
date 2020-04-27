"""Client-end for the ASR demo."""
import struct
import socket
import sys
import argparse
import wave

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--host_ip",
    default="localhost",
    type=str,
    help="Server IP address. (default: %(default)s)")
parser.add_argument(
    "--host_port",
    default=8086,
    type=int,
    help="Server Port. (default: %(default)s)")
parser.add_argument(
    "--fpath",
    default=None,
    type=str,
    help="Path to audio file. (default: %(default)s)")
args = parser.parse_args()


def main():

    received = None
    if args.fpath == None :
        print("Please specifiy audio file")
    else :

        audio = wave.open(args.fpath, 'rb')

        data_list = []

        for j in range(0, audio.getnframes()):
            current_frame = audio.readframes(1)
            data_list.append(current_frame)

        audio.close()

        if len(data_list) > 0:
            # Connect to server and send data
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((args.host_ip, args.host_port))
            sent = ''.join(data_list)
            sock.sendall(struct.pack('>i', len(sent)) + sent)
            print('Speech[length=%d] Sent.' % len(sent))
            # Receive data from the server and shut down
            received = sock.recv(1024)
            print("Recognition Results: %s" % (received))
            sock.close()
        else :
            print("Audio data is empty\n")

    
    return received



if __name__ == "__main__":
    main()
