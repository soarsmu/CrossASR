"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import json

from datetime import datetime

import argparse
import functools
import paddle.fluid as fluid
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from model_utils.model_check import check_cuda, check_version
from utils.error_rate import wer, cer
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_samples',      int,    10,     "# of samples to infer.")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('num_proc_bsearch', int,    8,      "# of CPUs for beam search.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  2.5,    "Coef of LM for beam search.")
add_arg('beta',             float,  0.3,    "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('infer_manifest',   str,
        'data/librispeech/manifest.dev-clean',
        "Filepath of manifest to infer.")
add_arg('mean_std_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'data/librispeech/vocab.txt',
        "Filepath of vocabulary.")
add_arg('lang_model_path',  str,
        'models/lm/common_crawl_00.prune01111.trie.klm',
        "Filepath for language model.")
add_arg('model_path',       str,
        './checkpoints/libri/step_final',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('decoding_method',  str,
        'ctc_beam_search',
        "Decoding method. Options: ctc_beam_search, ctc_greedy",
        choices = ['ctc_beam_search', 'ctc_greedy'])
add_arg('error_rate_type',  str,
        'wer',
        "Error rate type for evaluation.",
        choices=['wer', 'cer'])
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
# yapf: disable
args = parser.parse_args()



def infer():
    """Inference for DeepSpeech2."""

    """Start the ASR server"""
    # prepare data generator
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    data_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        keep_transcription_text=True,
        place=place,
        is_training=False)
    # prepare ASR model
    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        init_from_pretrained_model=args.model_path,
        place=place,
        share_rnn_weights=args.share_rnn_weights)

    # decoders only accept string encoded in utf-8
    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

    if args.decoding_method == "ctc_beam_search":
        ds2_model.init_ext_scorer(args.alpha, args.beta, args.lang_model_path,
                                  vocab_list)
    
    # prepare ASR inference handler
    def file_to_transcript(filename):
        feature = data_generator.process_utterance(filename, "")
        audio_len = feature[0].shape[1]
        mask_shape0 = (feature[0].shape[0] - 1) // 2 + 1
        mask_shape1 = (feature[0].shape[1] - 1) // 3 + 1
        mask_max_len = (audio_len - 1) // 3 + 1
        mask_ones = np.ones((mask_shape0, mask_shape1))
        mask_zeros = np.zeros((mask_shape0, mask_max_len - mask_shape1))
        mask = np.repeat(
            np.reshape(
                np.concatenate((mask_ones, mask_zeros), axis=1),
                (1, mask_shape0, mask_max_len)),
            32,
            axis=0)
        feature = (np.array([feature[0]]).astype('float32'),
                   None,
                   np.array([audio_len]).astype('int64').reshape([-1, 1]),
                   np.array([mask]).astype('float32'))
        probs_split = ds2_model.infer_batch_probs(
            infer_data=feature,
            feeding_dict=data_generator.feeding)

        if args.decoding_method == "ctc_greedy":
            result_transcript = ds2_model.decode_batch_greedy(
                probs_split=probs_split,
                vocab_list=vocab_list)
        else:
            result_transcript = ds2_model.decode_batch_beam_search(
                probs_split=probs_split,
                beam_alpha=args.alpha,
                beam_beta=args.beta,
                beam_size=args.beam_size,
                cutoff_prob=args.cutoff_prob,
                cutoff_top_n=args.cutoff_top_n,
                vocab_list=vocab_list,
                num_processes=1)
        return result_transcript[0]


    fdata = open(args.infer_manifest)
    data = fdata.readlines()
    fdata.close()


    file = open("sr_paddledeepspeech/execution_time/" + str(datetime.now()) + ".txt", "w+")


    result_transcripts = []
    i = 0
    for audio_data in data :
        i += 1
        start_time = time.time()
        filename = json.loads(audio_data)["audio_filepath"] 
        transcription = file_to_transcript(filename)
        print("DeepSpeech2 Translation - %d: %s" % (i, str(transcription)))
        result_transcripts.append(transcription)
        end_time = time.time()
        time_execution = round(end_time - start_time, 2)
        file.write("%d, %.2f\n" % (i, time_execution))

    file.close()

    paddledeepspeech_translation = open(
        "output/paddledeepspeech_translation_" + str(datetime.now()) + ".txt", "w+")
    
    i = 0
    for result in result_transcripts:
        i += 1
        # print("%d - %s" % (i, result))
        paddledeepspeech_translation.write("%s, %d, %s\n" %
                                            ("tts_google", i, result))
    paddledeepspeech_translation.close()




def main():
    print_arguments(args)
    infer()


if __name__ == '__main__':
    main()
