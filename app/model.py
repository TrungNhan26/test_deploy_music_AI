import numpy as np
from music21 import stream
import json
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from app.utils import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, concatenate, BatchNormalization as BatchNorm, Dense

def nhan_load_music_model():
    """
    Tải mô hình đã huấn luyện từ tệp .keras
    """
    MODEL_PATH = "app/saved_model/nhan_model.keras"
    model = load_model(MODEL_PATH)
    print(f"Đã tải mô hình từ {MODEL_PATH}")
    return model

def nhan_Melody_Generator(Note_Count=None, midi_file=None, length=40, mapping=None, reverse_mapping=None, X_seed=None, symb=None):
    """
    Sinh nhạc tự động dựa trên seed từ file MIDI hoặc ngẫu nhiên.
    """
    model = nhan_load_music_model()
    if midi_file:
        seed = midi_to_seed(midi_file, mapping, length)
    else:
        seed = X_seed[np.random.randint(0, len(X_seed) - 1)]
    
    Music = []
    Notes_Generated = []
    
    for i in range(Note_Count):
        seed = seed.reshape(1, length, 1)
        prediction = model.predict(seed, verbose=0)[0]
        if np.any(prediction <= 0):
            print("Warning: Prediction contains non-positive values. Adjusting...")
            prediction = np.log(np.maximum(prediction, 1e-7)) / 1.0  # diversity
        else:
            prediction = np.log(prediction) / 1.0  # diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index / float(len(symb))
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0], len(seed[0]), index_N)
        seed = seed[1:]
    
    Melody = chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)
    return Music, Melody_midi

def hoang_create_network(network_input_notes, n_vocab_notes, network_input_offsets, n_vocab_offsets, network_input_durations, n_vocab_durations):
    # Branch of the network that considers notes
    inputNotesLayer = Input(shape=(network_input_notes.shape[1], network_input_notes.shape[2]))
    inputNotes = LSTM(256, return_sequences=True)(inputNotesLayer)
    inputNotes = Dropout(0.2)(inputNotes)
    
    # Branch of the network that considers note offset
    inputOffsetsLayer = Input(shape=(network_input_offsets.shape[1], network_input_offsets.shape[2]))
    inputOffsets = LSTM(256, return_sequences=True)(inputOffsetsLayer)
    inputOffsets = Dropout(0.2)(inputOffsets)
    
    # Branch of the network that considers note duration
    inputDurationsLayer = Input(shape=(network_input_durations.shape[1], network_input_durations.shape[2]))
    inputDurations = LSTM(256, return_sequences=True)(inputDurationsLayer)
    inputDurations = Dropout(0.2)(inputDurations)
    
    # Concatenate the three input networks together
    inputs = concatenate([inputNotes, inputOffsets, inputDurations])
    
    # A cheeky LSTM to consider everything learned from the three separate branches
    x = LSTM(512, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(512)(x)
    x = BatchNorm()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    
    # Time to split into three branches again...
    
    # Branch of the network that classifies the note
    outputNotes = Dense(128, activation='relu')(x)
    outputNotes = BatchNorm()(outputNotes)
    outputNotes = Dropout(0.3)(outputNotes)
    outputNotes = Dense(n_vocab_notes, activation='softmax', name="Note")(outputNotes)
    
    # Branch of the network that classifies the note offset
    outputOffsets = Dense(128, activation='relu')(x)
    outputOffsets = BatchNorm()(outputOffsets)
    outputOffsets = Dropout(0.3)(outputOffsets)
    outputOffsets = Dense(n_vocab_offsets, activation='softmax', name="Offset")(outputOffsets)
    
    # Branch of the network that classifies the note duration
    outputDurations = Dense(128, activation='relu')(x)
    outputDurations = BatchNorm()(outputDurations)
    outputDurations = Dropout(0.3)(outputDurations)
    outputDurations = Dense(n_vocab_durations, activation='softmax', name="Duration")(outputDurations)
    
    # Create and compile the model
    model = Model(inputs=[inputNotesLayer, inputOffsetsLayer, inputDurationsLayer], outputs=[outputNotes, outputOffsets, outputDurations])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    model.load_weights('app/saved_model/weights-improvement-592-0.3251-bigger.hdf5')
    
    return model

def hoang_generate_notes(model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationnames, n_vocab_notes, n_vocab_offsets, n_vocab_durations):
    """ Generate notes from the neural network based on a sequence of notes """
    # Pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input_notes)-1)
    start2 = np.random.randint(0, len(network_input_offsets)-1)
    start3 = np.random.randint(0, len(network_input_durations)-1)

    int_to_note = {number: note for number, note in enumerate(notenames)}
    int_to_offset = {number: note for number, note in enumerate(offsetnames)}
    int_to_duration = {number: note for number, note in enumerate(durationnames)}

    pattern = network_input_notes[start]
    pattern2 = network_input_offsets[start2]
    pattern3 = network_input_durations[start3]
    prediction_output = []

    # Generate notes or chords
    for note_index in range(400):
        note_prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        note_prediction_input = note_prediction_input / float(n_vocab_notes)
        
        offset_prediction_input = np.reshape(pattern2, (1, len(pattern2), 1))
        offset_prediction_input = offset_prediction_input / float(n_vocab_offsets)
        
        duration_prediction_input = np.reshape(pattern3, (1, len(pattern3), 1))
        duration_prediction_input = duration_prediction_input / float(n_vocab_durations)

        prediction = model.predict([note_prediction_input, offset_prediction_input, duration_prediction_input], verbose=0)

        index = np.argmax(prediction[0])
        result = int_to_note[index]
        
        offset = np.argmax(prediction[1])
        offset_result = int_to_offset[offset]
        
        duration = np.argmax(prediction[2])
        duration_result = int_to_duration[duration]
        
        print(f"Next note: {result} - Duration: {duration_result} - Offset: {offset_result}")

        prediction_output.append([result, offset_result, duration_result])

        pattern.append(index)
        pattern2.append(offset)
        pattern3.append(duration)
        pattern = pattern[1:len(pattern)]
        pattern2 = pattern2[1:len(pattern2)]
        pattern3 = pattern3[1:len(pattern3)]

    return prediction_output

MAPPING_PATH = 'app/saved_model/mapping.json'
def roll_load_model_and_mapping(model_path="app/saved_model/roll_model.keras"):
    custom_objects = {
        'time_major': False,
        'learning_phase': 0
    }
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects
    )
    
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
        
    print("load model thành công ")
    
    return model, mappings

def roll_generate_melody(model, mappings, seed, num_steps, max_sequence_length, temperature, sequence_length):
    
    start_symbols = ["/"] * sequence_length
    seed = seed.split()
    melody = seed
    seed = start_symbols + seed
    seed = [mappings[symbol] for symbol in seed]
    
    
    for _ in range(num_steps):
        seed = seed[-max_sequence_length:]
        onehot_seed = keras.utils.to_categorical(seed, num_classes=len(mappings))
        onehot_seed = onehot_seed[np.newaxis, ...]
        
        probabilities = model.predict(onehot_seed)[0]
        output_int = roll_sample_with_temperature(probabilities, temperature)
        
        seed.append(output_int)
        
        output_symbol = [k for k, v in mappings.items() if v == output_int][0]
        
        if output_symbol == "/":
            break
        
        melody.append(output_symbol)
    
    return melody
