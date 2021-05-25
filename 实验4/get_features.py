# -*- coding: utf-8 -*-

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os

if __name__ == '__main__':
    save_dir = './imgs'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file = './data/chew.wav'
    x, sr = librosa.load(file, sr=22050)

    # 波形图 （Waveform）
    librosa.display.waveplot(x, sr=sr)

    tar = save_dir + '/wave.png'
    plt.savefig(tar)
    print('wave picture has been saved as file: \'{}\''.format(tar))
    plt.show()
    plt.cla()

    # 声谱图（spectrogram）
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram of aloe')

    tar = save_dir + '/spectrogram.png'
    plt.savefig(tar)
    print('spectrogram picture has been saved as file: \'{}\''.format(tar))
    plt.show()
    plt.cla()

    # 过零率 （Zero Crossing Rate）
    zero_crossings = librosa.zero_crossings(x, pad=False)
    print('ALL Zero Crossing Rate: ', sum(zero_crossings))
    limit = (len(x) - 200, len(x) - 100)
    zero_crossings = librosa.zero_crossings(x[limit[0]:limit[1]], pad=False)
    print('Zero Crossing Rate in range({0}, {1}): '.format(limit[0], limit[1]), sum(zero_crossings))

    # 频谱质心 （Spectral Centroid）
    spectral_centroids = librosa.feature.spectral_centroid(x, sr)[0]
    print('Spectral Centroid\'s shape: ', spectral_centroids.shape)
    # 计算时间变量
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    # 归一化频谱质心
    normalized = sklearn.preprocessing.minmax_scale(spectral_centroids, axis=0)
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalized, color='r')

    tar = save_dir + '/Spectral_Centroid.png'
    plt.savefig(tar)
    print('Spectral Centroid picture has been saved as file: \'{}\''.format(tar))
    plt.show()
    plt.cla()

    # 声谱衰减 (Spectral Roll-off）
    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr)[0]
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    normalized = sklearn.preprocessing.minmax_scale(spectral_rolloff, axis=0)
    plt.plot(t, normalized, color='b')

    tar = save_dir + '/Spectral_Roll-off.png'
    plt.savefig(tar)
    print('Spectral Roll-off picture has been saved as file: \'{}\''.format(tar))
    plt.show()
    plt.cla()

    # 色度频率 （Chroma Frequencies）
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma',
                             hop_length=hop_length, cmap='coolwarm')

    tar = save_dir + '/Chroma_Frequencies.png'
    plt.savefig(tar)
    print('Chroma Frequencies picture has been saved as file: \'{}\''.format(tar))
    plt.show()
    plt.cla()

    # MFCC特征提取 （ Mel Frequency Cepstral Coefficents ）
    mfccs = librosa.feature.mfcc(x, sr=sr)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    
    tar = save_dir + '/mfccs.png'
    plt.savefig(tar)
    print('mfcc feature picture has been saved as file: \'{}\''.format(tar))
    plt.show()
    plt.cla()
