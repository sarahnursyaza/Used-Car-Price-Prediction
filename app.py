from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json

gbr = joblib.load('model.pkl')

app = Flask(__name__, template_folder='template')


@app.route('/')
def home():
    return render_template('index.html')

def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(45)
    # set the numerical input as they are
    enc_input[0] = data['year']
    enc_input[1] = data['mileage_kms']
    enc_input[2] = data['power_hp']
    ##################### Make #########################
    # get the array of marks categories
    make = ['Mercedes-Benz', 'Mazda', 'Toyota', 'Honda', 'Volvo', 'Proton',
            'Lamborghini', 'BMW', 'Mitsubishi', 'Perodua', 'Ford',
            'Volkswagen', 'Nissan', 'Kia', 'MINI', 'Ferrari', 'Naza', 'Isuzu',
            'Suzuki', 'Hyundai', 'Land Rover', 'Audi', 'Lexus', 'Porsche',
            'Jaguar', 'Peugeot', 'Subaru', 'Chevrolet', 'Bentley', 'Daihatsu',
            'Ssangyong', 'Chery', 'Citroen', 'Smart', 'Renault', 'Maserati',
            'Infiniti', 'Rolls-Royce', 'Hummer', 'Inokom']
    cols = ['year', 'mileage_kms', 'power_hp', 'transmission_Automatic',
            'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
            'make_Chery', 'make_Chevrolet', 'make_Citroen', 'make_Daihatsu',
            'make_Ferrari', 'make_Ford', 'make_Honda', 'make_Hummer',
            'make_Hyundai', 'make_Infiniti', 'make_Inokom', 'make_Isuzu',
            'make_Jaguar', 'make_Kia', 'make_Lamborghini', 'make_Land Rover',
            'make_Lexus', 'make_MINI', 'make_Maserati', 'make_Mazda',
            'make_Mercedes-Benz', 'make_Mitsubishi', 'make_Naza', 'make_Nissan',
            'make_Perodua', 'make_Peugeot', 'make_Porsche', 'make_Proton',
            'make_Renault', 'make_Rolls-Royce', 'make_Smart', 'make_Ssangyong',
            'make_Subaru', 'make_Suzuki', 'make_Toyota', 'make_Volkswagen',
            'make_Volvo']

    # redefine the the user inout to match the column name
    redefinded_user_input = 'make_' + data['make']
    # search for the index in columns name list
    make_column_index = cols.index(redefinded_user_input)
    # print(make_column_index)
    # fullfill the found index with 1
    enc_input[make_column_index] = 1
    ##################### Transmision ####################
    # get the array of transmission
    transmission = ['Automatic', 'Manual']
    # redefine the the user inout to match the column name
    redefinded_user_input = 'transmission_' + data['transmission']
    # search for the index in columns name list
    transmission_column_index = cols.index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[transmission_column_index] = 1
    return enc_input


@app.route('/api', methods=['POST'])
def get_delay():
    result = request.form
    year = result['year']
    mileage_kms = result['mileage_kms']
    make = result['make']
    power_hp = result['power_hp']
    transmission = result['transmission']

    user_input = {'year': year, 'mileage_kms': mileage_kms, 'power_hp': power_hp, 'transmission': transmission,
                  'make': make}

    print(user_input)
    a = input_to_one_hot(user_input)
    price_pred = gbr.predict([a])[0]
    price_pred = round(price_pred, 2)
    #return json.dumps({'price': price_pred})
    return render_template('result.html',prediction=price_pred)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
