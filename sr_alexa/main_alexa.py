from alexa_client import AlexaClient
from alexa_client.alexa_client import constants
from alexa_client.alexa_client import helpers

import os, glob
import json

from time import sleep
from datetime import datetime

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
refresh_token = os.getenv("REFRESH_TOKEN")

BASE_URL_NORTH_AMERICA = 'alexa.na.gateway.devices.a2z.com'


if __name__ == '__main__':

    client = AlexaClient(
        client_id=client_id,
        secret=client_secret,
        refresh_token=refresh_token,
        base_url=BASE_URL_NORTH_AMERICA
    )

    client.connect()  # authenticate and other handshaking steps

    dialog_request_id = helpers.generate_unique_id()

    translation_writer = open("output/alexa_translation " + str(datetime.now()) + ".txt", "w+")

    data = "sr_alexa/alexa_data/"

    # for (dirpath, _, filenames) in os.walk(data):
    #     if (len(filenames) > 0):
    #         if not os.path.exists(dirpath):
    #             os.makedirs(dirpath)

    dirpath = "sr_alexa/alexa_data/tts_google/generated_speech/"
    for i in range(1, 20001):
        filename = "audio_" + str(i) + ".wav"
        fpath = os.path.join(dirpath, filename)
        print("Processing: " + fpath)
        audio = open(fpath, 'rb')
        try :
            directives = client.send_audio_file(
                audio, dialog_request_id=dialog_request_id)

            success = False
            text = ""
            if directives:
                for j, directive in enumerate(directives):
                    if directive.name == 'RenderTemplate':
                        payload = directive.payload
                        if ('textField' in payload.keys()) :
                            text = payload['textField']
                            print("Text: " + text)
                            success = True
            else:
                print("Audio " + str(i) + "- Can't get response")

            if (success):
                print("Transcription: " + text)
                translation_writer.write("%s\n" % (dirpath + ", " + filename[6:-4] + ", " + text))
            else :
                print("Transcription: ")
                translation_writer.write("%s\n" % (dirpath + ", " + filename[6:-4] + ", "))
                    
            sleep(1)

        except Exception as e :
            pass

    translation_writer.close()

