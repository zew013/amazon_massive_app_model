from flask import Flask, request
from flask_restful import Api, Resource
# import torch
import numpy as np
from transformers import BertTokenizer
from flask_cors import CORS
import json
#from json import JSONEncoder
# import pandas as pd
import io
import distill
import os
import onnxruntime as ort  # ONNX Runtime
import time

app = Flask(__name__)

CORS(app)
api = Api(app)


#model_path = './saved_model/model.pt'

#load_path = "https://amazonmassive.s3.us-west-1.amazonaws.com/model.pt"
load_path = "https://amazonmassive.s3.us-west-1.amazonaws.com/student.pt"
print(load_path)

#model = torch.load(model_path,map_location=torch.device('cpu'))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")

config = {
        'vocab_size' : len(tokenizer.get_vocab()),
        'embedding_size' : 300,
        'hidden_size' : 512,
        'fc_size' : 128,
        'num_layers' : 2,
        'n_classes' : 60,
        'dropout' : 0.5,
        'epochs' : 40,
        'lr' : 5e-4,
        'temp' : 1,
        'weight_decay' : 1e-4,
        'alpha' : 0.95,
        'batch_size' : 256,
        'input_dir' : 'assets',
        'dataset' : 'amazon',
        'ignore_cache' : False,
        'max_len' : 20,
        'early_stop' : 3
        #'output_dir' : 'result',
        }
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
config = AttributeDict(config)

# if not os.path.exists('local_model.pt'):
    
#     with smart_open(load_path, 'rb') as f:
#         with open('local_model.pt', 'wb') as local_f:
#             local_f.write(f.read())
#     print('downloaded model to local')
# update to use onnx
if not os.path.exists('student_model.onnx'):
    from smart_open import open as smart_open
    with smart_open(load_path, 'rb') as f:
        with open('student_model.onnx', 'wb') as local_f:
            local_f.write(f.read())
    print('Downloaded ONNX model to local')
ort_session = ort.InferenceSession("student_model.onnx")
# model = distill.StudentModel(config)
# model.load_state_dict(torch.load('local_model.pt', map_location=torch.device('cpu')))

# with smart_open(load_path, 'rb') as f:

#     #model=torch.load(io.BytesIO(f.read()),map_location=torch.device('cpu'))
#     model = distill.StudentModel(config)
#     model.load_state_dict(torch.load(io.BytesIO(f.read()),map_location=torch.device('cpu')))
#model = AutoModel.from_pretrained("cartesinus/bert-base-uncased-amazon-massive-intent")
#tokenizer = AutoTokenizer.from_pretrained("cartesinus/bert-base-uncased-amazon-massive-intent")
# argument parsing
#parser = reqparse.RequestParser()
#parser.add_argument('query')

INTENTS = [
    "datetime_query",
    "iot_hue_lightchange",
    "transport_ticket",
    "takeaway_query",
    "qa_stock",
    "general_greet",
    "recommendation_events",
    "music_dislikeness",
    "iot_wemo_off",
    "cooking_recipe",
    "qa_currency",
    "transport_traffic",
    "general_quirky",
    "weather_query",
    "audio_volume_up",
    "email_addcontact",
    "takeaway_order",
    "email_querycontact",
    "iot_hue_lightup",
    "recommendation_locations",
    "play_audiobook",
    "lists_createoradd",
    "news_query",
    "alarm_query",
    "iot_wemo_on",
    "general_joke",
    "qa_definition",
    "social_query",
    "music_settings",
    "audio_volume_other",
    "calendar_remove",
    "iot_hue_lightdim",
    "calendar_query",
    "email_sendemail",
    "iot_cleaning",
    "audio_volume_down",
    "play_radio",
    "cooking_query",
    "datetime_convert",
    "qa_maths",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "transport_query",
    "music_likeness",
    "email_query",
    "play_music",
    "audio_volume_mute",
    "social_post",
    "alarm_set",
    "qa_factoid",
    "calendar_set",
    "play_game",
    "alarm_remove",
    "lists_remove",
    "transport_taxi",
    "recommendation_movies",
    "iot_coffee",
    "music_query",
    "play_podcasts",
    "lists_query",
]

class status(Resource):
    def get(self):
        try:
            return {'data':'Api running'}
        except(error):
            return {'data':error}

class PredictIntent(Resource):
    def get(self, order):
        # use parser and find the user's query
        start_time = time.time()

        # vectorize the user's query and make a prediction
        # tokenized = tokenizer([str(order)], padding='max_length', truncation=True, max_length=20)
        # for key, value in tokenized.items():
        #     tokenized[key] = torch.tensor(value)
        tokenized = tokenizer(
            [str(order)],
            padding='max_length',
            truncation=True,
            max_length=config.max_len,
            return_tensors='np'  # Use NumPy tensors for ONNX Runtime
        )
        ort_inputs = {
            'input_ids': tokenized['input_ids'].astype(np.int64),
            'attention_mask': None
        }
        logits = ort_session.run(None, ort_inputs)[0]
        # Apply softmax to get probabilities
        s = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        s = s[0]  # Since batch size is 1
        #s = torch.nn.functional.softmax(model(tokenized, None )).detach().numpy()[0]

        # model(tokenized['input_ids'], None )
        # s = torch.nn.functional.softmax(model(tokenized['input_ids'], None )).detach().numpy()[0]

        index = np.argsort(s)[-5:][::-1]
        prob = s[index]
        intent = np.array(INTENTS)[index]

        result_list = [{"intent": str(intent[i]), "prob": round(float(prob[i]), 3)} for i in range(len(intent))]
        result_json = json.dumps(result_list)
        # print(result_json)
        # Serialization if want to display np array
        #numpyData = {"array": numpyArrayOne}
        #encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
        # create JSON object
        end_time = time.time()

        output = {
                'prediction': str(intent[0]), 
                'confidence': str(prob[0]), 
                'df': result_json, # pd.DataFrame({'intent':intent, 'prob':prob}).to_json(orient = 'records')
                'time_infer': end_time - start_time
                }
        # print(pd.DataFrame({'intent':intent, 'prob':prob}).to_json(orient = 'records'))
        # curl -X GET 'http://127.0.0.1:5001/PredictIntent/tell%me%the%time'
        # curl -X GET 'http://127.0.0.1:5001/PredictIntent/tell%me%the%time%please%please%tell%me%the%time%please%pleastell%me%the%time%please%pleastell%me%the%time%please%pleas'
        return output

# http://127.0.0.1:5001/PredictIntent/tell me the time

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(status, '/')
api.add_resource(PredictIntent, '/PredictIntent/<string:order>')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)