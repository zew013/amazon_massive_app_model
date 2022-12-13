from flask import Flask, request
from flask_restful import Api, Resource
import torch
import numpy as np
from transformers import BertTokenizer
from flask_cors import CORS
import json
from json import JSONEncoder
import pandas as pd
app = Flask(__name__)

CORS(app)
api = Api(app)


model_path = './saved_model/model.pt'
model = torch.load(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")

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

class PredictIntent(Resource):
    def get(self, order):
        # use parser and find the user's query


        # vectorize the user's query and make a prediction
        tokenized = tokenizer([str(order)], padding='max_length', truncation=True, max_length=20)
        for key, value in tokenized.items():
            tokenized[key] = torch.tensor(value)

        s = torch.nn.functional.softmax(model(tokenized, None )).detach().numpy()[0]

        
        index = np.argsort(s)[-5:][::-1]
        prob = s[index]
        intent = np.array(INTENTS)[index]

        # Serialization if want to display np array
        #numpyData = {"array": numpyArrayOne}
        #encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
        # create JSON object
        output = {
                'prediction': str(intent[0]), 
                'confidence': str(prob[0]), 
                'df': pd.DataFrame({'intent':intent, 'prob':prob}).to_json(orient = 'records')
                }

        return output
# http://127.0.0.1:5000/PredictIntent/tell me the time

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictIntent, '/PredictIntent/<string:order>')


if __name__ == '__main__':
    app.run()