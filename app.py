import torch
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = whisper.load_model("large")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    audio = model_inputs.get('audio', None)
    if audio == None:
        return {'message': "No input provided"}
   
    language = model_inputs.get('language', None)
    no_speech_threshold = model_inputs.get('no_speech_threshold', 0.1)
    logprob_threshold = model_inputs.get('logprob_threshold', -1.0)

    args = {
        "language": language,
        "logprob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
    }

    mp3Bytes = BytesIO(base64.b64decode(audio.encode("ISO-8859-1")))
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    # Run the model
    result = model.transcribe("input.mp3", **args)
    output = {"text":result["text"],"language":language}
    print(output)
    os.remove("input.mp3")
    # Return the results as a dictionary
    return output
