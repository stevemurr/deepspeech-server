import argparse
import bjoern
from deepspeech.model import Model
import falcon
import json
import logging
import numpy as np
import os
import subprocess
import sys
import time


class SpeechRecognitionResource(object):
    def __init__(self, ds):
        self.ds = ds

    def on_get(self, req, res):
        pass

    def on_post(self, req, res):
        """ on_post accepts a file param containing an audio file """

        body = req.get_param('file')
        if body is not None:
            body = body.file
        else:
            body = req.stream
        audio, fs = convert_samplerate(body.read())        
        result = self.ds.stt(audio, fs)

        res.set_header("Access-Control-Allow-Origin", "*")
        res.status = falcon.HTTP_200
        res.media = result


def convert_samplerate(body):
    """ convert_samplerate converts raw audio input to 16 bit 16khz mono audio """
    sox_cmd = 'sox - --type raw --bits 16 --channels 1 --rate 16000 - '
    try:
        p = subprocess.Popen(sox_cmd.split(),
                             stderr=subprocess.PIPE, 
                             stdout=subprocess.PIPE, 
                             stdin=subprocess.PIPE)
        output, err = p.communicate(input=body)

        if p.returncode:
            raise RuntimeError('SoX returned non-zero status: {}'.format(err))

    except OSError as e:
        raise OSError('SoX not found, use 16kHz files or install it: ', e)

    audio = np.fromstring(output, dtype=np.int16)
    return audio, 16000


def setup_args():
    """ setup_args parses the arguments defined and loads the params file 

    # Return
        args: Parsed arguments.
        params: Parsed params.json file. 
        err: Optional error
    """
    parser = argparse.ArgumentParser(description='DeepSpeech SR server.')
    parser.add_argument('--port', type=int, default=8080, help="Port to host server on.")
    parser.add_argument('--params', default="params.json", help="Path to json configuration file.")

    args = parser.parse_args()
    if not os.path.exists(args.params):
        return None, None, "params file not found" 

    with open(args.params, "r") as f:
        params = json.load(f)

    return args, params, None


def setup_api(routes, middleware=None):
    """ setup_api sets up the api given the routes 
    
    # Arguments
        routes: Dictionary containing {"/path/to/resource": ResourceObject } 
        middleware: Array of middleware objects. 
    # Return
        api: Falcon API object.
    """
    api = falcon.API(middleware=middleware)
    for key, value in routes.items():
        api.add_route(key, value)

    return api


def setup_logger():
    """ setup_logger returns a middleware logging object for use with the falcon API """
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    class ResponseLoggerMiddleware(object):
        def __init__(self, logger):
            self.logger = logger

        def process_response(self, req, resp, resource, req_succeeded):
            t = time.gmtime(time.time())
            curtime = "[{0}-{1}-{2} {3}:{4}:{5}]".format(t[0], t[1], t[2], t[3], t[4], t[5])
            self.logger.info('{0} [{1} {2} {3}] {4}'.format(
                curtime,
                req.method,
                req.relative_uri,
                resp.status[:3],
                resp.media))

    return ResponseLoggerMiddleware(logger)


def check_err(err, fn=None):
    """ check_err handles error objects 

    # Arguments
        err: Error object.
        fn: Optional custom function handler. 
    """
    if err:
        print(err)
        sys.exit(-1)
    fn(err)


if __name__ == '__main__':
    args, params, err = setup_args()
    if err:
        check_err(err)

    ds = Model(
        params["model"], 
        params["n_features"], 
        params["n_context"], 
        params["alphabet"], 
        params["beam_width"])
    ds.enableDecoderWithLM(
        params["alphabet"], 
        params["lm"], 
        params["trie"], 
        params["lm_weight"], 
        params["word_count_weight"], 
        params["valid_word_count_weight"])

    logger = setup_logger()
    routes = {
        "/api/reco": SpeechRecognitionResource(ds)
    } 
    
    api = setup_api(routes, middleware=logger)

    try:
        bjoern.run(api, host='0.0.0.0', port=args.port)
    except KeyboardInterrupt:
        sys.exit(0)
