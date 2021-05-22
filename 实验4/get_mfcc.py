# -*- coding: utf-8 -*-

import librosa
import librosa.display
import matplotlib.pyplot as plt

if __name__ == '__main__':
    save = True

    file = './data/chew.wav'
    x, sr = librosa.load(file, sr=22050)
    mfccs = librosa.feature.mfcc(x, sr)

    print('shape: ', mfccs.shape)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')

    if save:
        save_file = './mfcc_feature.png'
        plt.savefig(save_file)
        print('feature picture saved as file: \'{}\''.format(save_file))

    plt.show()
