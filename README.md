# DeepSpeech Server

DeepSpeech SR server using Falcon.

# Usage

Install the dependencies  
`pip install -r requirements.txt`

If you want GPU acceleration  
`pip install deepspeech-gpu --upgrade`

Download the pre-trained model  
`wget -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.1.1/deepspeech-0.1.1-models.tar.gz | tar xvfz -`

Update path to model files in the `params.json` file.  

Run the server  
`python main.py --params params.json`

- SpeechRecognition: `POST /api/reco`

# Params

Parameters for the service are stored in a JSON file.  The follow parameters control the beam search decoder.

`beam_width` Used in the CTC decoder when building candidate transcription.   
`lm_weight` Hyperparameter for the CTC decoder.  Language model weight.  
`word_count_weight` Hyperparameter of the CTC decoder.  Word insertion weight (penalty).  
`valid_word_count_weight` Valid word insertion weight.  This is used to lessen the word insertion penalty.  
`n_features` Number of MFCC features to use.  
`n_context` Size of the context window used for producing timesteps in the input vector.  

# Examples

`http -f POST http://localhost:8080/api/reco < audio_file.wav`

