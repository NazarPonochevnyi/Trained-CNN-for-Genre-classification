import imp
import librosa
import numpy as np
from keras.models import load_model

genres = {0: "metal", 1: "disco", 2: "classical", 3: "hiphop", 4: "jazz", 
            5: "country", 6: "pop", 7: "blues", 8: "reggae", 9: "rock"}
song_samples = 660000

def load_song(filepath):
    y, sr = librosa.load(filepath)
    y = y[:song_samples]
    return y, sr

def splitsongs(X, window = 0.1, overlap = 0.5):
    temp_X = []

    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))

    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)

    return np.array(temp_X)

def to_melspec(signals):
    melspec = lambda x : librosa.feature.melspectrogram(x, n_fft=1024, hop_length=512)[:, :, np.newaxis]
    spec_array = map(melspec, signals)
    return np.array(list(spec_array))
       
def get_genre(path, debug=False):
    model = load_model('genres_full_vgg16.h5')
    
    y = load_song(path)[0]
    predictions = []
    spectro = []
    signals = splitsongs(y)
    spec_array = to_melspec(signals)
    spectro.extend(spec_array)
    spectro = np.array(spectro)
    spectro = np.squeeze(np.stack((spectro,)*3,-1))

    pr = np.array(model.predict(spectro))
    predictions = np.argmax(pr, axis=1)
    if debug:
        print('Load audio:', path)
        print("\nFull Predictions:")
        for p in pr: print(list(p))
        print("\nPredictions:\n{}".format(predictions))
        print("Confidences:\n{}".format([round(x, 2) for x in np.amax(pr, axis=1)]))
        print("\nOutput Predictions:\n{}\nPredicted class:".format(np.mean(pr, axis=0)))
    
    return genres[np.bincount(predictions).argmax()] # list(np.mean(pr, axis=0))

if __name__ == '__main__':
    print(get_genre('./../../../../../LSTM-Music-Genre-Classification/audios/reggae_music.mp3', True))
