import os
from app.model import *
from app.utils import *
import pickle

def generate():
    """Generate a piano MIDI file."""
    
    # Load the notes, durations, and offsets used to train the model
    with open('app/saved_model/Notes.txt', 'rb') as filepath:
        notes = pickle.load(filepath)
    
    with open('app/saved_model/Durations.txt', 'rb') as filepath:
        durations = pickle.load(filepath)
    
    with open('app/saved_model/Offset.txt', 'rb') as filepath:
        offsets = pickle.load(filepath)
    
    # Create sorted sets and calculate vocab sizes
    notenames = sorted(set(item for item in notes))
    n_vocab_notes = len(set(notes))
    network_input_notes, normalized_input_notes = hoang_prepare_sequences(notes, notenames, n_vocab_notes)
    
    offsetnames = sorted(set(item for item in offsets))
    n_vocab_offsets = len(set(offsets))
    network_input_offsets, normalized_input_offsets = hoang_prepare_sequences(offsets, offsetnames, n_vocab_offsets)
    
    durationnames = sorted(set(item for item in durations))
    n_vocab_durations = len(set(durations))
    network_input_durations, normalized_input_durations = hoang_prepare_sequences(durations, durationnames, n_vocab_durations)
    
    # Create the neural network model
    model = hoang_create_network(normalized_input_notes, n_vocab_notes, 
                           normalized_input_offsets, n_vocab_offsets, 
                           normalized_input_durations, n_vocab_durations)
    
    # Generate notes using the trained model
    prediction_output = hoang_generate_notes(
        model, 
        network_input_notes, 
        network_input_offsets, 
        network_input_durations, 
        notenames, 
        offsetnames, 
        durationnames, 
        n_vocab_notes, 
        n_vocab_offsets, 
        n_vocab_durations
    )
    
    return prediction_output


def nhan_generate_melody(midi_url ,Note_Count):
    # Tải các dữ liệu cần thiết
    corpus_content = load_corpus('app/saved_model/corpus.txt')
    symbols = load_symbols('app/saved_model/symb.txt')
    X_seed = load_x_seed('app/saved_model/X_seed.txt')
    mapping = load_mapping('app/saved_model/mapping.txt')
    reverse_mapping = load_reverse_mapping('app/saved_model/reverse_mapping.txt')
    
    # Sinh melody từ file MIDI gốc
    Music, Melody_midi = nhan_Melody_Generator(Note_Count, midi_file=midi_url, length=40,
                                           mapping=mapping, reverse_mapping=reverse_mapping,
                                           X_seed=X_seed, symb=symbols)

    return Music,Melody_midi  # Trả về đường dẫn đầy đủ của file MIDI đã lưu

