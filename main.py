#!/usr/bin/env python

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.
NOTE: This module requires the dependencies `pyaudio` and `termcolor`.
To install using pip:
    pip install pyaudio
    pip install termcolor
Example usage:
    python3 main.py
"""

# [START speech_transcribe_infinite_streaming]

import re
import sys
import time

#Speech-to-text
from google.cloud import speech
import pyaudio
from six.moves import queue

# Haystack for search and NLP
from haystack.document_store import ElasticsearchDocumentStore
from nlp_search import preprocess_docs

from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store import ElasticsearchDocumentStore
# import text converters and preprocessors
from haystack.file_converter.txt import TextConverter
from haystack.file_converter.pdf import PDFToTextConverter
from haystack.file_converter.docx import DocxToTextConverter
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.retriever.dense import DensePassageRetriever
from haystack.pipeline import ExtractiveQAPipeline, DocumentSearchPipeline

#Translation
from google.cloud import translate

# You can use platform-specific text-to-speech engines, or the open source ESpeakNG
#from espeakng import ESpeakNG

# For Mac
#import mac_say
from mac_say.gtts import say

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Parameters for translation
PROJECT_ID="gm-demos"
TRANSLATION_REQ=True
SOURCE="it"
TARGET="en"

# Parameters for document processing
#DOC_DIR="immigration_docs"
DOC_DIR="medical_docs"

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"

# Test or production
TESTING=True

# translate api client
def translate_text(text="YOUR_TEXT_TO_TRANSLATE", project_id="YOUR_PROJECT_ID", source="it", target="en"):
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": source,
            "target_language_code": target,
        }
    )

    # Display the translation for each input text provided
    #for translation in response.translations:
    #    print("Translated text: {}".format(translation.translated_text))
    
    # return translation.translated_text
    return response.translations[0].translated_text



def get_current_time():
    """Return Current Time in ms."""

    return int(round(time.time() * 1000))

# use local microphone audio stream
class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self):

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):

        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""

        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Stream Audio from microphone to API and to local buffer"""

        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)

# listen to input, retrieve answers, loop until exit
def listen_print_loop(responses, stream, pipe):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """

    for response in responses:

        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (STREAMING_LIMIT * stream.restart_counter)
        )
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:

            sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            if TRANSLATION_REQ:
                print(
                    translate_text(text=transcript, project_id=PROJECT_ID, source=SOURCE, target=TARGET)
                )

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            # keyword for stop in target language, e.g. "stop" in English
            if re.search(r"\b(exit|stop)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                break
            # keyword for search in source language, e.g. "ricerca" in Italian
            elif re.search(r"\b(ricerca|cerca)\b", transcript, re.I):
                print ("Query=", previous_transcript)
                if TRANSLATION_REQ:
                    # translate the query
                    translated_query=translate_text(text=previous_transcript, project_id=PROJECT_ID, source=SOURCE, target=TARGET)
                    print("Translated query= ", translated_query)
                    previous_transcript=translated_query
                #stream.closed=True
                # retrieve 4 documents via search, infer 2 answers with NLP
                answers = pipe.run(query=previous_transcript, params={"Retriever": {"top_k": 4}, "Reader": {"top_k": 2}})
                # translate and read back the answers
                short_answer(answers)
                
                sys.stdout.write(YELLOW)
                print("\nNEW REQUEST\n")

            
            previous_transcript=transcript


        else:
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")

            stream.last_transcript_was_final = False

#translate and read back
def short_answer(answers):
    for i, answer in enumerate(answers['answers']):
        print('Answer no. {}: {}, score: {}%'.format(i, answer['answer'], round(answer['score'] * 100)))
        if TRANSLATION_REQ:
            #esng = ESpeakNG()
            #esng.voice=SOURCE
            # reverse translation of answers
            translated_answer=translate_text(text=answer['answer'], project_id=PROJECT_ID, source=TARGET, target=SOURCE)
            print (translated_answer)
            #esng.say(translated_answer)
            #mac_say.say(["-v", "Alice", translated_answer]) #Alice is the Italian voice
            say(SOURCE, translated_answer)
            #time.sleep(15)

def main():

    warnings.filterwarnings('ignore')
    print("\nInitializing document store, retriever, reader and audio streaming. Please wait...\n")

    # Prime document store
    document_store = ElasticsearchDocumentStore()

    if TESTING:
        # run in experimental phase to clean up everything
        document_store.delete_all_documents()


        # Load fresh documents into doc store
        #preprocess_docs(DOC_DIR)

        all_docs = convert_files_to_dicts(dir_path=DOC_DIR)
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_by="word",
            split_length=100,
            split_respect_sentence_boundary=True,
            split_overlap=10
        )
        docs = preprocessor.process(all_docs)

        print(f"n_files_input: {len(all_docs)}\nn_docs_output: {len(docs)}")

        document_store.write_documents(docs)

    # Prepare the extractive pipe

    retriever = DensePassageRetriever(document_store=document_store,
                                    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                    max_seq_len_query=64,
                                    max_seq_len_passage=256,
                                    batch_size=16,
                                    use_gpu=True,
                                    embed_title=True,
                                    use_fast_tokenizers=True)

    if TESTING: document_store.update_embeddings(retriever)

    # Best recall, but requires gpu for decent processing times
    #reader = FARMReader(model_name_or_path="ahotrod/albert_xxlargev1_squad2_512", use_gpu=True)

    # Decent compromise between recall and proc time
    #reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    reader = FARMReader(model_name_or_path="ahotrod/electra_large_discriminator_squad2_512", use_gpu=True)



    pipe = ExtractiveQAPipeline(reader, retriever)

    #answers = pipe.run(query="what are the rights of immigrants?", params={"Retriever": {"top_k":5}, "Reader": {"top_k": 2}})
    #short_answer(answers)

    """start bidirectional streaming from microphone input to speech API"""

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=SOURCE,
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    with mic_manager as stream:

        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write(
                "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )

            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream, pipe)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write("\n")
            stream.new_stream = True


if __name__ == "__main__":

    main()

# [END speech_transcribe_infinite_streaming]