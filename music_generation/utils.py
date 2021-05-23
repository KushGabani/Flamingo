import os
from datetime import datetime

import music21
import numpy as np

def get_data(artist):
    print("-----------------Loading compressed dataset")
    data = np.load(f"music_generation/data/{artist}_processed.npz")
    x = data['arr_0']

    print("-----------------Loading personal artist pitches")
    data = np.load(f"music_generation/data/{artist}_pitchnames.npz")
    pitchnames = data['arr_0']
    n_vocab = len(pitchnames)
    return x, pitchnames, n_vocab

def generate_sequence(model, notes, x, pitchnames, n_vocab):
    start = np.random.randint(0, len(x) - 1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = x[start]
    prediction_output = []
    print("-----------------generating new music patterns")
    for ni in range(notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]
    return prediction_output

def generate_notes(predictions):
    offset = 0
    output_notes = []

    print("-----------------converting patterns to notes and chords")
    for pattern in predictions:
        # If the generated pattern is a chord, then we'll have to split the array of generated
        # notes, convert them into sounds and again put it back together in a Chord object
        if pattern.isdigit() or '.' in pattern:
            notes_in_chord = pattern.split(".")
            notes= []

            for current_note in notes_in_chord:
                new_note = music21.note.Note(int(current_note))
                new_note.storedInstrument = music21.instrument.Piano()
                notes.append(new_note)
            new_chord = music21.chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        
        # If the generated pattern is a note, then store them in a Note object
        else:
            new_note = music21.note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = music21.instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5
    return output_notes

def save_midi(music, artist):
    print("-----------------Saving generated music")
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    midi_stream = music21.stream.Stream(music)
    filename = f'created_music/{artist}-{timestamp}.mid'
    midi_stream.write("midi", fp=f'static/{filename}')
    print(f"-----------------Generate music saved: static/{filename}")

    return filename