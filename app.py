import pickle
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, url_for, render_template

#from src.nnmodels import NeuralNetwork

class NeuralNetwork(torch.nn.Module):
    def __init__(self: 'NeuralNetwork') -> None:
        """
        Initializes the neural network by creating the layers of the network.
        """
        super(NeuralNetwork,self).__init__()
        
        # Create the layers of the neural network
        self.linear1 = torch.nn.Linear(in_features=152, out_features=200)
        self.linear2 = torch.nn.Linear(in_features=200, out_features=100)
        self.linear3 = torch.nn.Linear(in_features=100, out_features=64)
        self.linear4 = torch.nn.Linear(in_features=64, out_features=64)
        self.output = torch.nn.Linear(in_features=64, out_features=1)
    
    def forward(self: 'NeuralNetwork',x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input Tensor with shape (#batches, #rows, #columns)

        Returns:
            torch.Tensor: output predicted value (#batches, #rows, 1)
        """
        # Propagate the input tensor through the layers of the neural network
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.nn.functional.relu(self.linear4(x))
        x = self.output(x)
        return x

app = Flask(__name__)

# Load the Model and assets
nn_model = torch.load('models/sadapala_assignment0_part_3.pickle')
label_encoders = pickle.load(open('models/part_3_label_encoders.pickle','rb'))
transformers = pickle.load(open('models/part_3_transformers.pickle','rb'))

# Initializations
input_features = ['Name', 'Measure', 'Measure Info', 'Geo Type Name', 'Geo Place Name', 'Season']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    x = pd.DataFrame([data])[input_features]
    # Label Encoding
    for feature in input_features:
        x[feature] = label_encoders[feature].transform(x[feature])
    
    # One Hot Encoding
    final_input_features = input_features.copy()
    for feature in input_features:
        x = transformers[feature].transform(x)
        final_input_features.remove(feature)
        final_input_features = list(transformers[feature].transformers_[0][1].get_feature_names_out()) + final_input_features
        
        if type(x) == np.ndarray:
            x = pd.DataFrame(x, columns = final_input_features)
        else:
            x = pd.DataFrame(x.toarray(), columns = final_input_features)
    
    
    x = torch.tensor(x.values, requires_grad=False).float()
    
    nn_model.train(False)
    with torch.inference_mode():
        y_hat = nn_model(x)
    
    return jsonify(y_hat.item())

@app.route('/predict', methods=['POST'])
def predict():
    data={}
    
    for key in request.form.keys():
        data[key] = request.form[key]
    
    x = pd.DataFrame([data])[input_features]
    # Label Encoding
    for feature in input_features:
        x[feature] = label_encoders[feature].transform(x[feature])
    
    # One Hot Encoding
    final_input_features = input_features.copy()
    for feature in input_features:
        x = transformers[feature].transform(x)
        final_input_features.remove(feature)
        final_input_features = list(transformers[feature].transformers_[0][1].get_feature_names_out()) + final_input_features
        
        if type(x) == np.ndarray:
            x = pd.DataFrame(x, columns = final_input_features)
        else:
            x = pd.DataFrame(x.toarray(), columns = final_input_features)
    
    
    x = torch.tensor(x.values, requires_grad=False).float()
    
    nn_model.train(False)
    with torch.inference_mode():
        y_hat = nn_model(x)
    
    return render_template('home.html', prediction_text = 'the predicted air quality is: {0}'.format(y_hat.item()))

if __name__ == '__main__':
    app.run(debug=True)
    
    
    