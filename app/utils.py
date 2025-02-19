from music21 import instrument,note, chord, stream, converter
import music21 as m21
import numpy as np
from datetime import datetime
import json
from fastapi.responses import FileResponse
import os
from app.database import *

def load_corpus(file_path):
    """
    Hàm tải nội dung của một file văn bản vào chuỗi.
    Parameters:
        file_path (str): Đường dẫn đến file cần tải.
    Returns:
        str: Nội dung file dưới dạng một chuỗi.
    """
    try:
        with open(file_path, 'r') as corpus_file:
            corpus = corpus_file.read()
        return corpus
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_symbols(file_path):
    """
    Hàm tải danh sách các ký hiệu từ file văn bản, mỗi dòng là một ký hiệu.
    Parameters:
        file_path (str): Đường dẫn đến file cần tải.
    Returns:
        list: Danh sách các ký hiệu được đọc từ file.
    """
    try:
        with open(file_path, 'r') as symb_file:
            symbols = [line.strip() for line in symb_file.readlines()]
        return symbols
    except Exception as e:
        print(f"Error: {e}")
        return []

def load_x_seed(file_path, reshape_dims=(-1, 40, 1)):
    """
    Hàm tải dữ liệu X_seed từ file văn bản và khôi phục lại shape ban đầu.
    Parameters:
        file_path (str): Đường dẫn đến file chứa dữ liệu.
        reshape_dims (tuple): Kích thước mong muốn sau khi reshape.
    Returns:
        np.ndarray: Mảng numpy đã được khôi phục kích thước.
    """
    try:
        x_seed = np.loadtxt(file_path)
        x_seed = x_seed.reshape(reshape_dims)
        return x_seed
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_y_seed(file_path):
    """
    Hàm tải dữ liệu y_seed từ file văn bản và khôi phục lại shape ban đầu nếu cần.
    Parameters:
        file_path (str): Đường dẫn đến file chứa dữ liệu.
    Returns:
        np.ndarray: Mảng numpy chứa dữ liệu y_seed đã được reshape.
    """
    try:
        y_seed = np.loadtxt(file_path)
        y_seed = y_seed.reshape(-1, y_seed.shape[1])  # Đảm bảo giữ shape ban đầu
        return y_seed
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_mapping(file_path):
    """
    Hàm tải mapping từ file JSON.
    Parameters:
        file_path (str): Đường dẫn đến file chứa mapping.
    Returns:
        dict: Dữ liệu mapping được đọc từ file.
    """
    try:
        with open(file_path, 'r') as f:
            mapping = json.load(f)
        return mapping
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file '{file_path}'")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_reverse_mapping(file_path):
    """
    Hàm tải reverse_mapping từ file JSON và chuyển các key thành kiểu integer.
    Parameters:
        file_path (str): Đường dẫn đến file chứa reverse_mapping.
    Returns:
        dict: Reverse mapping với các key được chuyển thành integer.
    """
    try:
        with open(file_path, 'r') as f:
            reverse_mapping = json.load(f)
            # Chuyển các key thành integer
            reverse_mapping_converted = {int(key): value for key, value in reverse_mapping.items()}
        return reverse_mapping_converted
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file '{file_path}'")
        return None
    except ValueError:
        print(f"Error: Failed to convert keys to integers in file '{file_path}'")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def midi_to_seed(midi_file, mapping, length=40):
    """
    Đọc một file MIDI và chuyển đổi thành seed cho mô hình với chiều dài mặc định là 40.
    """
    midi_data = converter.parse(midi_file)
    notes = []

    for element in midi_data.recurse():
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    # Ánh xạ các nốt và hợp âm thành số nguyên dựa trên `mapping`
    seed = [mapping[n] for n in notes if n in mapping]

    # Điều chỉnh seed để khớp với chiều dài mặc định `length`
    if len(seed) > length:
        seed = seed[-length:]
    else:
        seed = [0] * (length - len(seed)) + seed

    return np.reshape(seed, (1, length, 1)) / float(len(mapping))

def chords_n_notes(music_sequence):
    """
    Chuyển đổi danh sách nốt nhạc và hợp âm thành đối tượng music21
    """
    melody = []
    for item in music_sequence:
        if '.' in item or item.isdigit():
            notes_in_chord = item.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(notes)
            melody.append(new_chord)
        else:
            new_note = note.Note(item)
            melody.append(new_note)
    return melody

def hoang_prepare_sequences(notes, pitchnames, n_vocab):
    """ 
    Prepare the sequences used by the Neural Network 
    """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def hoang_create_midi(prediction_output_all, username):
    """ Convert the output from the prediction to notes and create a MIDI file from the notes. """
    offset = 0
    output_notes = []

    # Initialize lists to store note data
    offsets = []
    durations = []
    notes = []

    # Extract notes, durations, and offsets from the prediction output
    for x in prediction_output_all:
        notes = np.append(notes, x[0])  # Store notes
        try:
            offsets = np.append(offsets, float(x[1]))  # Store offsets (start time)
        except:
            num, denom = x[1].split('/')  # Handle fractions in offset
            x[1] = float(num) / float(denom)
            offsets = np.append(offsets, float(x[1]))

        durations = np.append(durations, x[2])  # Store durations
    print("Creating MIDI File...")

    # Create note and chord objects based on the values generated by the model
    x = 0  # Counter for accessing note and chord data
    for pattern in notes:
        # Check if the pattern is a chord (i.e., contains a period or digits)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')  # Split into individual notes
            chord_notes = [note.Note(int(current_note)) for current_note in notes_in_chord]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()

            new_chord = chord.Chord(chord_notes)

            try:
                new_chord.duration.quarterLength = float(durations[x])
            except:
                # Handle fraction duration
                num, denom = durations[x].split('/')
                new_chord.duration.quarterLength = float(num) / float(denom)

            new_chord.offset = offset
            output_notes.append(new_chord)  # Append chord to output

        # If it's just a single note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()

            try:
                new_note.duration.quarterLength = float(durations[x])
            except:
                # Handle fraction duration for notes
                num, denom = durations[x].split('/')
                new_note.duration.quarterLength = float(num) / float(denom)

            output_notes.append(new_note)  # Append note to output

        # Increase the offset to avoid overlapping notes
        try:
            offset += offsets[x]
        except:
            # Handle fraction offset
            num, denom = offsets[x].split('/')
            offset += float(num) / float(denom)

        x += 1  # Move to the next note/chord

        # Tạo thư mục "results" nếu chưa tồn tại
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Tìm giá trị i lớn nhất trong các file đã tồn tại
    max_index = 0
    for file_name in os.listdir(results_dir):
        if file_name.startswith(username) and file_name.endswith(".midi"):
            try:
                # Lấy phần số i từ tên file
                index_part = file_name.split("_no_midi_multitone_")[1].split(".midi")[0]
                index = int(index_part)
                max_index = max(max_index, index)
            except (IndexError, ValueError):
                pass  # Bỏ qua các file không hợp lệ

    # Tăng giá trị i để tạo tên file mới
    new_index = max_index + 1
    file_name = f"{username}_no_midi_multitone_{new_index}.midi"
    title = file_name.replace(".midi", "")
    full_url = title
    output_file_path = os.path.join(results_dir, file_name)
    # Create a music21 Stream object from the output notes
    midi_stream = stream.Stream(output_notes)
    # Save MIDI to the 'results/' directory with the formatted file name
    midi_stream.write('midi', fp=output_file_path)
    update_record("musics", "title", title, "title = 'Clone title'")
    update_record("musics", "full_url", full_url, "full_url = 'http://example.com/xxx'")
    print("MIDI file created successfully!")
    
    # Convert file path to URL-friendly format (replace backslashes with forward slashes)
    file_url = output_file_path.replace(os.sep, "/")

    return file_url, file_name

def roll_prepare_seed(seed, mappings, sequence_length):
    seed = seed.split()
    seed = ["/"] * sequence_length + seed
    seed = [mappings[symbol] for symbol in seed]
    return seed

def roll_sample_with_temperature(probabilities, temperature):
    predictions = np.log(probabilities) / temperature
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
    
    choices = range(len(probabilities))
    index = np.random.choice(choices, p=probabilities)
    
    return index    

def roll_save_melody_without_midi(melody, composer_username, step_duration=0.25):
    # Tạo thư mục "results" nếu chưa tồn tại
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Tìm giá trị i lớn nhất trong các file đã tồn tại
    max_index = 0
    for file_name in os.listdir(results_dir):
        if file_name.startswith(composer_username) and file_name.endswith(".midi"):
            try:
                # Lấy phần số i từ tên file
                index_part = file_name.split("_no_midi_monotone_")[1].split(".midi")[0]
                index = int(index_part)
                max_index = max(max_index, index)
            except (IndexError, ValueError):
                pass  # Bỏ qua các file không hợp lệ

    # Tăng giá trị i để tạo tên file mới
    new_index = max_index + 1
    file_name = f"{composer_username}_no_midi_monotone_{new_index}.midi"
    title = file_name.replace(".midi", "")
    full_url =   title
    file_path = os.path.join(results_dir, file_name)

    # Xử lý lưu melody vào file MIDI
    stream = m21.stream.Stream()
    start_symbol = None
    step_counter = 1

    for i, symbol in enumerate(melody):
        if symbol != "_" or i + 1 == len(melody):
            if start_symbol is not None:
                quarter_length_duration = step_duration * step_counter
                if start_symbol == "r":
                    m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                else:
                    m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                stream.append(m21_event)
                step_counter = 1
            start_symbol = symbol
        else:
            step_counter += 1

    # Lưu file MIDI
    stream.write("midi", file_path)
    print(f"Melody saved to {file_path}")
    update_record("musics", "title", title, "title = 'Clone title'")
    update_record("musics", "full_url", full_url, "full_url = 'http://example.com/xxx'")
    # Chuyển đổi dấu \ thành /
    file_url = file_path.replace(os.sep, "/")

    # Trả về tên file và đường dẫn
    return file_name, file_url

def roll_encode_song(song, time_step=0.25):
    encoded_song = []
    for event in song.flatten().notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = str(event.pitch.midi)
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        else:
            continue  # Bỏ qua nếu không phải nốt hoặc nghỉ
        steps = max(1, int(event.duration.quarterLength / time_step))
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    return " ".join(encoded_song)



def roll_save_melody_with_midi(melody,composer_username, step_duration=0.20):
        # Tạo thư mục "results" nếu chưa tồn tại
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Tìm giá trị i lớn nhất trong các file đã tồn tại
    max_index = 0
    for file_name in os.listdir(results_dir):
        if file_name.startswith(composer_username) and file_name.endswith(".midi"):
            try:
                # Lấy phần số i từ tên file
                index_part = file_name.split("_yes_midi_monotone_")[1].split(".midi")[0]
                index = int(index_part)
                max_index = max(max_index, index)
            except (IndexError, ValueError):
                pass  # Bỏ qua các file không hợp lệ

    # Tăng giá trị i để tạo tên file mới
    new_index = max_index + 1
    file_name = f"{composer_username}_yes_midi_monotone_{new_index}.midi"
    title = file_name.replace(".midi", "")
    full_url = title
    file_path = os.path.join(results_dir, file_name)

    # Xử lý lưu melody vào file MIDI
    stream = m21.stream.Stream()
    start_symbol = None
    step_counter = 1

    for i, symbol in enumerate(melody):
        if symbol != "_" or i + 1 == len(melody):
            if start_symbol is not None:
                quarter_length_duration = step_duration * step_counter
                if start_symbol == "r":
                    m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                else:
                    m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                stream.append(m21_event)
                step_counter = 1
            start_symbol = symbol
        else:
            step_counter += 1

    stream.write("midi", file_path)
    print(f"Melody saved to {file_path}")
    update_record("musics", "title", title, "title = 'Clone title'")
    update_record("musics", "full_url", full_url, "full_url = 'http://example.com/xxx'")
    # Chuyển đổi dấu \ thành /
    file_url = file_path.replace(os.sep, "/")

    # Trả về tên file và đường dẫn
    return file_name, file_url

def rename_file_in_results(old_filename, new_filename):
    folder_path = "results"

    # Nếu old_filename không có phần mở rộng, tự động thêm .midi
    if not old_filename.endswith(".midi"):
        old_filename += ".midi"
    if not new_filename.endswith(".midi"):
        new_filename += ".midi"

    old_path = os.path.join(folder_path, old_filename)
    new_path = os.path.join(folder_path, new_filename)

    if not os.path.exists(old_path):
        print(f"Tệp '{old_filename}' không tồn tại trong thư mục '{folder_path}'.")
        return False

    os.rename(old_path, new_path)
    print(f"Đã đổi tên: {old_filename} ➝ {new_filename}")
    return True

