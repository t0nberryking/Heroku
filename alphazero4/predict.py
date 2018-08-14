from flask import Flask
from flask import request
from flask import jsonify
import json


import numpy as np
from game import GameState, Game
import config
from model import Residual_CNN
from agent import Agent
import pickle


app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            gs = GameState(np.array(json.loads(data["gameState"])), 1)
            #print(gs)
            
            current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + (6,7), 42, config.HIDDEN_CNN_LAYERS)
            current_NN.model.set_weights(current_NN.read('connect4', 2, 74).get_weights())
            current_player = Agent('current_player', 84, 42, config.MCTS_SIMS, config.CPUCT, current_NN)

            preds = current_player.get_preds(gs)
            preds = np.array(preds[1]).reshape(6,7)
            pred_arg = np.unravel_index(preds.argmax(), preds.shape)
            
        except ValueError:
            return jsonify("Please enter a number.")

        return jsonify([int(x) for x in pred_arg])

if __name__ == '__main__':
    app.run(debug=True)
