from flask import Flask, Response, request, jsonify
import joblib
from flask_cors import CORS
import serial
import time
import random
import numpy as np

# Load the trained model
model = joblib.load(r'C:/Users/MSI/Desktop/LED_Panel_Project/LED_Panel/trained_model.joblib')


app = Flask(__name__)
CORS(app)

ser_in = None
ser_out = None

try:
    ser_in = serial.Serial("COM6", 9600)
    print("Serial connection to COM6 established")
except serial.SerialException as e:
    print(f"Error opening serial port COM6: {e}")

try:
    ser_out = serial.Serial("COM12", 9600)
    print("Serial connection to COM12 established")
except serial.SerialException as e:
    print(f"Error opening serial port COM12: {e}")

current_level = 1
received_numbers = []
stream_stop_event = False
max_attempts = 5

keypad_map = {
    '1': 15, '2': 14, '3': 13, 'A': 12,
    '4': 11, '5': 10, '6': 9, 'B': 8,
    '7': 7, '8': 6, '9': 5, 'C': 4,
    '*': 3, '0': 2, '#': 1, 'D': 0
}

symbol_values = ["0", "#", "*", "+", "X"]

def has_vertical_triplet(array, triplet_indexes):
    for i in range(12):
        if i + 8 <= 15:
            if array[i] == array[i + 4] == array[i + 8]:
                if not (i in triplet_indexes and i + 4 in triplet_indexes and i + 8 in triplet_indexes):
                    return True
    return False

def generate_random_combination():
    while True:
        triplet_index = random.randint(0, 7)
        triplet_value = random.randint(0, 4)
        display_array = [random.randint(0, 4) for _ in range(16)]
        display_array[triplet_index] = triplet_value
        display_array[triplet_index + 4] = triplet_value
        display_array[triplet_index + 8] = triplet_value
        triplet_indexes = [triplet_index, triplet_index + 4, triplet_index + 8]
        if not has_vertical_triplet(display_array, triplet_indexes):
            return display_array, triplet_indexes

def send_combination_to_arduino(display_array):
    data_str = ','.join(map(str, display_array)) + '\n'
    try:
        ser_out.write(data_str.encode('utf-8'))
        print(f"Sent to COM12: {data_str}")
    except serial.SerialException as e:
        print(f"Error writing to serial port COM12: {e}")

def clear_serial_buffer():
    if ser_in:
        ser_in.reset_input_buffer()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input parameters from the request
        data = request.json
        
        # Extract features from the request
        features = data.get('features')
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Ensure features is a list of numerical values
        if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
            return jsonify({'error': 'Features must be a list of numerical values'}), 400
        
        # Convert features to the appropriate format
        features = np.array(features).reshape(1, -1)
        
        # Check the shape of the features
        expected_shape = (1, model.n_features_in_)  # Example: Adjust if necessary
        if features.shape != expected_shape:
            return jsonify({'error': f'Expected features with shape {expected_shape}, got {features.shape}'}), 400
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = int(prediction[0])
        
        # Return the prediction result
        return jsonify({'prediction': predicted_class}), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
        

@app.route('/change-level', methods=['POST'])
def change_level():
    global current_level, stream_stop_event
    data = request.json
    level = data.get('level')
    if level in [1, 2, 3]:
        # Stop the current stream and set the new level
        stream_stop_event = True
        time.sleep(1)  # Allow time for the current stream to stop
        stream_stop_event = False
        current_level = level
        clear_serial_buffer()
        print(f"Changed to level {current_level}")
        return '', 200
    return 'Invalid level', 400

def event_stream_level1():
    global stream_stop_event, max_attempts
    attempts = 0
    while not stream_stop_event:
        display_array, triplet_indexes = generate_random_combination()
        send_combination_to_arduino(display_array)
        received_numbers.clear()
        while len(received_numbers) < 3 and attempts < max_attempts:
            try:
                if ser_in.in_waiting > 0:
                    data = ser_in.readline().decode('utf-8').strip()
                    print(f"Data received from serial COM11: {data}")
                    if data in keypad_map:
                        received_numbers.append(keypad_map[data])
                        if len(received_numbers) == 3:
                            break
                time.sleep(0.1)
            except serial.SerialException as e:
                print(f"Error reading from serial port COM11: {e}")
                break

        triplet_indexes.sort()
        received_numbers.sort()
        print(f"Received numbers: {received_numbers}, Expected triplet: {triplet_indexes}")
        
        if received_numbers == triplet_indexes:
            print("Correctly matched")
            yield f"data: Correct\n\n"
        else:
            print("Wrong match")
            yield f"data: Wrong\n\n"
        
        attempts += 1
        if attempts >= max_attempts:
            yield f"data: Max attempts reached\n\n"
            break

        time.sleep(1)
        if not stream_stop_event:
            display_array, triplet_indexes = generate_random_combination()
            send_combination_to_arduino(display_array)

def event_stream_level2():
    global stream_stop_event, max_attempts
    attempts = 0
    while not stream_stop_event:
        display_array, _ = generate_random_combination()
        send_combination_to_arduino(display_array)
        symbol_to_find_index = random.randint(0, 4)
        symbol_to_find = symbol_values[symbol_to_find_index]
        symbol_value = symbol_to_find_index
        symbol_positions = [i for i, x in enumerate(display_array) if x == symbol_value]
        yield f"data: Symbol to find: {symbol_to_find}\n\n"
        received_numbers.clear()
        symbol_positions_set = set(symbol_positions)
        print(f"Display Array: {display_array}")
        print(f"Expected positions for symbol '{symbol_to_find}' (value {symbol_value}): {symbol_positions}")
        
        while not stream_stop_event and attempts < max_attempts:
            try:
                if ser_in.in_waiting > 0:
                    data = ser_in.readline().decode('utf-8').strip()
                    print(f"Data received from serial COM11: {data}")
                    if data in keypad_map:
                        pos = keypad_map[data]
                        print(f"Checking position: {pos}")
                        if pos in symbol_positions_set:
                            if pos not in received_numbers:
                                received_numbers.append(pos)
                                print(f"Received valid position: {pos}")
                        else:
                            print(f"Wrong position pressed: {pos}")
                            yield f"data: Wrong position pressed: {pos}\n\n"
                            break
                if set(received_numbers) == symbol_positions_set:
                    yield f"data: Correct\n\n"
                    break
                time.sleep(3)
            except serial.SerialException as e:
                print(f"Error reading from serial port COM11: {e}")
                break
        
        attempts += 1
        if attempts >= max_attempts:
            yield f"data: Max attempts reached\n\n"
            break

def event_stream_level3():
    global stream_stop_event, max_attempts
    attempts = 0
    while not stream_stop_event:
        display_array = [random.randint(0, 4) for _ in range(16)]
        
        # Generate a random vertical pattern with unique symbols
        pattern = []
        while len(pattern) < 3:
            symbol = random.randint(0, 4)
            if symbol not in pattern:
                pattern.append(symbol)
        
        # Ensure the pattern is vertical and placed from top to bottom
        while True:
            pattern_index = random.randint(0, 7)  # Only up to index 7 to fit vertical triplet
            for i in range(3):
                display_array[pattern_index + i * 4] = pattern[i]
            if display_array[pattern_index] != display_array[pattern_index + 4] and display_array[pattern_index + 4] != display_array[pattern_index + 8]:
                break
        
        send_combination_to_arduino(display_array)
        received_numbers.clear()
        
        # Reverse the pattern and pattern positions for correct display order
        reversed_pattern = pattern[::-1]
        reversed_pattern_symbols = [symbol_values[val] for val in reversed_pattern]
        pattern_positions = [pattern_index + i * 4 for i in range(3)]
        reversed_pattern_positions = pattern_positions[::-1]
        
        # Send the reversed pattern symbols to the client
        yield f"data: Pattern to find: {reversed_pattern_symbols}\n\n"
        
        pattern_positions_set = set(reversed_pattern_positions)
        
        print(f"Display Array: {display_array}")
        print(f"Pattern symbols: {reversed_pattern_symbols}")
        print(f"Pattern positions: {reversed_pattern_positions}")
        
        while not stream_stop_event and attempts < max_attempts:
            try:
                if ser_in.in_waiting > 0:
                    data = ser_in.readline().decode('utf-8').strip()
                    print(f"Data received from serial COM11: {data}")
                    if data in keypad_map:
                        pos = keypad_map[data]
                        print(f"Checking position: {pos}")
                        if pos in pattern_positions_set:
                            if pos not in received_numbers:
                                received_numbers.append(pos)
                                print(f"Received valid position: {pos}")
                        else:
                            print(f"Wrong position pressed: {pos}")
                            yield f"data: Wrong position pressed: {pos}\n\n"
                            break
                if set(received_numbers) == pattern_positions_set:
                    yield f"data: Correct\n\n"
                    break
                time.sleep(0.1)
            except serial.SerialException as e:
                print(f"Error reading from serial port COM11: {e}")
                break
        
        attempts += 1
        if attempts >= max_attempts:
            yield f"data: Max attempts reached\n\n"
            break

@app.route('/stream/level1')
def stream_level1():
    global stream_stop_event
    stream_stop_event = False
    return Response(event_stream_level1(), mimetype="text/event-stream")

@app.route('/stream/level2')
def stream_level2():
    global stream_stop_event
    stream_stop_event = False
    return Response(event_stream_level2(), mimetype="text/event-stream")

@app.route('/stream/level3')
def stream_level3():
    global stream_stop_event
    stream_stop_event = False
    return Response(event_stream_level3(), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
