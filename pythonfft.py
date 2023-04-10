#coding:utf-8
import csv
import wave
import struct
from scipy import fromstring, int16
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def fourier(x, n, w):
    K = []
    for i in range(0, w):
        sample = x[i * n:(i+1) * n]
        partial = np.fft.fft(sample)
        K.append(partial)

    return K

def binarydata(arr,threshold):
    arr_binary = arr.copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] > threshold:
                arr_binary[i][j] = 1
            else:
                arr_binary[i][j] = 0
    return arr_binary

def fftdata(fn, ln, threshold): #fn:セッション ln:ラベル threshold:閾値

    if fn == 1:
        foldername = '1_dataset' #フォルダ名

    elif fn == 2:
        foldername = '2_dataset' #フォルダ名

    elif fn == 3:
        foldername = '3_dataset' #フォルダ名

    elif fn == 4:
        foldername = '4_dataset' #フォルダ名

    elif fn == 5:
        foldername = '5_dataset' #フォルダ名


    """
    if ln == 0:
        filename = str(fn) + '_super_shaver_0'#ファイル名
        labelnumber = 0 #ラベル

    elif ln == 1:
        filename = str(fn) + '_radicon_1'#ファイル名
        labelnumber = 1 #ラベル

    elif ln == 2:
        filename = str(fn) + '_shredder_2'#ファイル名
        labelnumber = 2 #ラベル

    elif ln == 3:
        filename = str(fn) + '_mixer_3'#ファイル名
        labelnumber = 3 #ラベル

    elif ln == 4:
        filename = str(fn) + '_drilldriver_4'#ファイル名
        labelnumber = 4 #ラベル

    elif ln == 5:
        filename = str(fn) + '_ultrabrush_5'#ファイル名
        labelnumber = 5 #ラベル

    elif ln == 6:
        filename = str(fn) + '_handymassager_6'#ファイル名
        labelnumber = 6 #ラベル
    """

    if ln == 0:
        filename = str(fn) + '_super_shaver'#ファイル名
        labelnumber = 0 #ラベル

    elif ln == 1:
        filename = str(fn) + '_shredder'#ファイル名
        labelnumber = 1 #ラベル

    elif ln == 2:
        filename = str(fn) + '_mixer'#ファイル名
        labelnumber = 2 #ラベル

    elif ln == 3:
        filename = str(fn) + '_drilldriver'#ファイル名
        labelnumber = 3 #ラベル

    elif ln == 4:
        filename = str(fn) + '_ultrabrush'#ファイル名
        labelnumber = 4 #ラベル


    wavfile = filename + '.wav'
    wr = wave.open(wavfile, "rb")
    ch = wr.getnchannels()
    width = wr.getsampwidth()
    fr = wr.getframerate()
    fn = wr.getnframes()

    N = 96 #1ms間のフレーム数
    span = 500 #0.5秒
    st = 1.0 * N * span / fr

    """
    print('チャンネル', ch)
    print('サンプル幅', width)
    print('総フレーム数', fn)
    print('サンプル時間', st, '秒')
    """
    origin = wr.readframes(wr.getnframes()) #全フレーム読み込み
    data = origin[:N * span * ch * width]
    wr.close()

    """
    print('現配列長', len(origin))
    print('サンプル配列長: ', len(data))
    """
    if width == 2:
        X = np.frombuffer(data, dtype="int16")
    elif width == 4:
        X = np.frombuffer(data, dtype="int32")

    #print(len(X))

    #閾値設定してcsvデータ生成


    K = fourier(X, N, span)
    arr_K = np.array(K)
    arr_K = arr_K[:, 0:48]
    freqlist = np.fft.fftfreq(N, d=1/fr)
    arr_freqlist = freqlist[0:48]
    amp = []
    amp_binary = []

    print(arr_freqlist)
    
    for i in range(len(arr_K)): 
        a = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in arr_K[i]]
        amp.append(a)
        b = []
        for j in a:
            if j >= threshold: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        amp_binary.append(b)


    #閾値を周波数帯毎に設定
    """
    for i in range(len(arr_K)): 
        a = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in arr_K[i]]
        amp.append(a)
        b = []
        for j in range(0,5):
            if a[j] >= 170000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(5,10):
            if a[j] >= 30000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(10,15):
            if a[j] >= 19000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(15,20):
            if a[j] >= 8000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(20,25):
            if a[j] >= 6000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(25,30):
            if a[j] >= 5000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(30,48):
            if a[j] >= 4000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        amp_binary.append(b)
    """
    
    """
    for i in range(len(arr_K)): 
        a = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in arr_K[i]]
        amp.append(a)
        b = []
        for j in range(0,5):
            if a[j] >= 150000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(5,10):
            if a[j] >= 30000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(10,15):
            if a[j] >= 15000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(15,20):
            if a[j] >= 8000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(20,30):
            if a[j] >= 5000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        for j in range(30,48):
            if a[j] >= 4000: #閾値 40kHzの単周波音の振幅が5000越え，-10dB（200mV）で振幅1000
                b.append(1)
            else:
                b.append(0)
        amp_binary.append(b)
    """

   #センサアレイの再現バイナリデータ

    arr_amp = np.array(amp)
    arr_amp_binary = np.array(amp_binary)
    arr_freq = np.array(arr_freqlist)
    ultra_amp = arr_amp[:, 20:48]
    low_amp = arr_amp[:, 0:27]
    arduino_amp = arr_amp[:, 0:5]
    ultra_amp_binary = arr_amp_binary[:, 20:48]
    low_amp_binary = arr_amp_binary[:, 0:27]
    ultra_freq = arr_freq[20:48]
    low_freq = arr_freq[0:27]

    all_amp_binary1 = arr_amp_binary[:, 0:3] #周波数0から2000Hz
    all_amp_binary2 = arr_amp_binary[:, 3:6]
    all_amp_binary3 = arr_amp_binary[:, 6:9]
    all_amp_binary4 = arr_amp_binary[:, 9:12]
    all_amp_binary5 = arr_amp_binary[:, 12:15]
    all_amp_binary6 = arr_amp_binary[:, 15:18]
    all_amp_binary7 = arr_amp_binary[:, 18:21]
    all_amp_binary8 = arr_amp_binary[:, 21:24]
    all_amp_binary9 = arr_amp_binary[:, 24:27]
    all_amp_binary10 = arr_amp_binary[:, 27:30]
    all_amp_binary11 = arr_amp_binary[:, 30:33]
    all_amp_binary12 = arr_amp_binary[:, 33:36]
    all_amp_binary13 = arr_amp_binary[:, 36:39]
    all_amp_binary14 = arr_amp_binary[:, 39:42]
    all_amp_binary15 = arr_amp_binary[:, 42:45]
    all_amp_binary16 = arr_amp_binary[:, 45:48] #周波数45000から47000Hz

    #周波数帯域のバイナリの値を合算

    all_sum1 = all_amp_binary1.sum(1)
    all_sum2 = all_amp_binary2.sum(1)
    all_sum3 = all_amp_binary3.sum(1)
    all_sum4 = all_amp_binary4.sum(1)
    all_sum5 = all_amp_binary5.sum(1)
    all_sum6 = all_amp_binary6.sum(1)
    all_sum7 = all_amp_binary7.sum(1)
    all_sum8 = all_amp_binary8.sum(1)
    all_sum9 = all_amp_binary9.sum(1)
    all_sum10 = all_amp_binary10.sum(1)
    all_sum11 = all_amp_binary11.sum(1)
    all_sum12 = all_amp_binary12.sum(1)
    all_sum13 = all_amp_binary13.sum(1)
    all_sum14 = all_amp_binary14.sum(1)
    all_sum15 = all_amp_binary15.sum(1)
    all_sum16 = all_amp_binary16.sum(1)


    lowsensor_amp_binary1 = low_amp_binary[:, 4:7] #周波数4000から6000Hz
    lowsensor_amp_binary2 = low_amp_binary[:, 9:12] #周波数9000から11000Hz
    lowsensor_amp_binary3 = low_amp_binary[:, 14:17] #周波数14000から16000Hz
    lowsensor_amp_binary4 = low_amp_binary[:, 19:22] #周波数19000から21000Hz
    lowsensor_amp_binary5 = low_amp_binary[:, 24:27] #周波数24000から26000Hz

    lowsensor_sum1 = lowsensor_amp_binary1.sum(1)
    lowsensor_sum2 = lowsensor_amp_binary2.sum(1)
    lowsensor_sum3 = lowsensor_amp_binary3.sum(1)
    lowsensor_sum4 = lowsensor_amp_binary4.sum(1)
    lowsensor_sum5 = lowsensor_amp_binary5.sum(1)

    ultrasensor_amp_binary1 = ultra_amp_binary[:, 5:8] #周波数25000から27000Hz
    ultrasensor_amp_binary2 = ultra_amp_binary[:, 6:10] #周波数26000から29000Hz
    ultrasensor_amp_binary3 = ultra_amp_binary[:, 9:12] #周波数29000から31000Hz
    ultrasensor_amp_binary4 = ultra_amp_binary[:, 11:15] #周波数31000から34000Hz
    ultrasensor_amp_binary5 = ultra_amp_binary[:, 19:22] #周波数39000から41000Hz

    ultrasensor_sum1 = ultrasensor_amp_binary1.sum(1)
    ultrasensor_sum2 = ultrasensor_amp_binary2.sum(1)
    ultrasensor_sum3 = ultrasensor_amp_binary3.sum(1)
    ultrasensor_sum4 = ultrasensor_amp_binary4.sum(1)
    ultrasensor_sum5 = ultrasensor_amp_binary5.sum(1)

    all_sum = np.zeros((len(arr_amp), 16))
    ultrasensor_sum = np.zeros((len(ultra_amp), 5))
    lowsensor_sum = np.zeros((len(low_amp), 5))


    for j in range(len(arr_amp)):
        all_sum[j][0] = all_sum1[j]
        all_sum[j][1] = all_sum2[j]
        all_sum[j][2] = all_sum3[j]
        all_sum[j][3] = all_sum4[j]
        all_sum[j][4] = all_sum5[j]
        all_sum[j][5] = all_sum6[j]
        all_sum[j][6] = all_sum7[j]
        all_sum[j][7] = all_sum8[j]
        all_sum[j][8] = all_sum9[j]
        all_sum[j][9] = all_sum10[j]
        all_sum[j][10] = all_sum11[j]
        all_sum[j][11] = all_sum12[j]
        all_sum[j][12] = all_sum13[j]
        all_sum[j][13] = all_sum14[j]
        all_sum[j][14] = all_sum15[j]
        all_sum[j][15] = all_sum16[j]

    for j in range(len(ultra_amp)):

        ultrasensor_sum[j][0] = ultrasensor_sum1[j]
        ultrasensor_sum[j][1] = ultrasensor_sum2[j]
        ultrasensor_sum[j][2] = ultrasensor_sum3[j]
        ultrasensor_sum[j][3] = ultrasensor_sum4[j]
        ultrasensor_sum[j][4] = ultrasensor_sum5[j]
    for j in range(len(low_amp)):

        lowsensor_sum[j][0] = lowsensor_sum1[j]
        lowsensor_sum[j][1] = lowsensor_sum2[j]
        lowsensor_sum[j][2] = lowsensor_sum3[j]
        lowsensor_sum[j][3] = lowsensor_sum4[j]
        lowsensor_sum[j][4] = lowsensor_sum5[j]

    all_sum_binary = all_sum.copy()
    ultrasensor_sum_binary = ultrasensor_sum.copy()
    lowsensor_sum_binary = lowsensor_sum.copy()

    #合算した値をバイナリデータに修正

    for i in range(len(all_sum)):
        for j in range(16):
            if all_sum[i][j] > 0:
                all_sum_binary[i][j] = 1
            else:
                all_sum_binary[i][j] = 0

    for i in range(len(ultrasensor_sum)):
        for j in range(5):
            if ultrasensor_sum[i][j] > 0:
                ultrasensor_sum_binary[i][j] = 1
            else:
                ultrasensor_sum_binary[i][j] = 0

    for i in range(len(lowsensor_sum)):
        for j in range(5):
            if lowsensor_sum[i][j] > 0:
                lowsensor_sum_binary[i][j] = 1
            else:
                lowsensor_sum_binary[i][j] = 0

    arr_amp_label = np.insert(arr_amp, 48, labelnumber, axis=1)
    ultra_amp_label = np.insert(ultra_amp, 28, labelnumber, axis=1)
    arduino_amp_label = np.insert(arduino_amp, 5, labelnumber, axis=1)
    ultrasensor_sum_binary_label = np.insert(ultrasensor_sum_binary, 5, labelnumber, axis=1)
    lowsensor_sum_binary_label = np.insert(lowsensor_sum_binary, 5, labelnumber, axis=1)
    arr_amp_all_binary_label = np.insert(arr_amp_binary, 48, labelnumber, axis=1)
    all_sum_binary_label = np.insert(all_sum_binary, 16, labelnumber, axis=1)

    print(arr_amp_label)
    print(ultra_amp_label)
    print(arduino_amp_label)
    print(ultrasensor_sum_binary_label)
    print(lowsensor_sum_binary_label)
    print(arr_amp_all_binary_label)
    print(all_sum_binary_label)

    #全ての周波数帯の振幅データ(1msFFT)

    np.savetxt('C:/Users/sohei/source/repos/pythonfft/pythonfft/'+foldername+'/' +filename+ '_05s.csv', arr_amp_label, delimiter=',')

    #超音波帯の振幅データ(1msFFT)

    np.savetxt('C:/Users/sohei/source/repos/pythonfft/pythonfft/'+foldername+'_ultra/' +filename + '_05s_ultra.csv', ultra_amp_label, delimiter=',')

    #8000HzFFTの振幅データ（1msFFT）

    np.savetxt('C:/Users/sohei/source/repos/pythonfft/pythonfft/'+foldername+'_8000/'+filename + '_05s_8000.csv', arduino_amp_label, delimiter=',')

    #実際のセンサのピーク周波数帯のバイナリデータ（仮想データ）

    np.savetxt('C:/Users/sohei/source/repos/pythonfft/pythonfft/'+foldername+'_virtual/'+filename + '_05s_virtual_' + str(threshold) + '.csv', ultrasensor_sum_binary_label, delimiter=',', fmt='%d')

    #低周波数帯のバイナリデータ（仮想データ）

    np.savetxt('C:/Users/sohei/source/repos/pythonfft/pythonfft/'+foldername+'_lowvirtual/'+filename + '_05s_lowvirtual_' + str(threshold) + '.csv', lowsensor_sum_binary_label, delimiter=',', fmt='%d')
    #全ての周波数帯のバイナリデータ

    np.savetxt('C:/Users/sohei/source/repos/pythonfft/pythonfft/'+foldername+'_allb/'+filename + '_05s_allb_' + str(threshold) + '.csv', all_sum_binary_label, delimiter=',', fmt='%d')


for i in range(1,6):
    for j in range(0,5):
        fftdata(i, j, 4000)

#全体FFT
"""
wr = wave.open('1_mixer_3.wav', "rb")
ch = wr.getnchannels()
width = wr.getsampwidth()
fr = wr.getframerate()
fn = wr.getnframes()
print(width)

N = 96
span = 500
st = 1.0 * N * span / fr

origin = wr.readframes(wr.getnframes()) #全フレーム読み込み
data = origin[:N * span * ch * width]
wr.close()


if width == 2:
    X = np.frombuffer(data, dtype="int16")
elif width == 4:
    X = np.frombuffer(data, dtype="int32")
"""
"""
F = np.fft.fft(X)
length = len(F)
freq = np.fft.fftfreq(length, d=1/fr)
print(freq)
fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,6))
ax[0].plot(F.real, label="Real part")
ax[0].legend()
ax[1].plot(F.imag, label="Imaginary part")
ax[1].legend()
ax[2].plot(freq, label="Frequency")
ax[2].legend()
ax[2].set_xlabel("Number of data")
plt.show()

Amp = np.abs(F)
fig, ax = plt.subplots()
ax.plot(freq[1:int(length/2)], Amp[1:int(length/2)])
ax.set_xlabel("Freqency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()
"""


#時間波形
"""

time = np.arange(0, st, st/len(X))
plt.plot(time, X)
plt.show()
"""
#1msecの時間波形
"""
x0 = X[950:1046]
time0 = np.arange(0.01, 0.011, 0.001/96)
plt.plot(time0, x0)
plt.show()
"""

#2000個目の周波数スペクトル
"""
K = fourier(X, N, span)
arr_K = np.array(K)
arr_K = arr_K[:, 0:48]
freqlist = np.fft.fftfreq(N, d=1/fr)
arr_freqlist = freqlist[0:48]
#amp = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in arr_K[2000]]
#plt.plot(arr_freqlist[1:], amp[1:])
#plt.show()
fullamp=[]
for i in range(len(arr_K)): 
    a = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in arr_K[i]]
    fullamp.append(a)
arr_fullamp = np.array(fullamp)
ultra_fullamp = arr_fullamp[:, 20:48]
ultra_fullamp_1 = ultra_fullamp[:, 5:8] #周波数25000から27000Hz
ultra_fullamp_2 = ultra_fullamp[:, 6:9] #周波数26000から28000Hz
ultra_fullamp_3 = ultra_fullamp[:, 9:12] #周波数29000から31000Hz
ultra_fullamp_4 = ultra_fullamp[:, 14:17] #周波数34000から36000Hz
ultra_fullamp_5 = ultra_fullamp[:, 19:22] #周波数39000から41000Hz

ultra_fullamp_binary_1 = binarydata(ultra_fullamp_1, 50000)
ultra_fullamp_binary_2 = binarydata(ultra_fullamp_2, 50000)
ultra_fullamp_binary_3 = binarydata(ultra_fullamp_3, 50000)
ultra_fullamp_binary_4 = binarydata(ultra_fullamp_4, 5000)
ultra_fullamp_binary_5 = binarydata(ultra_fullamp_5, 5000)

ultrasensor_sum1 = ultra_fullamp_binary_1.sum(1)
ultrasensor_sum2 = ultra_fullamp_binary_2.sum(1)
ultrasensor_sum3 = ultra_fullamp_binary_3.sum(1)
ultrasensor_sum4 = ultra_fullamp_binary_4.sum(1)
ultrasensor_sum5 = ultra_fullamp_binary_5.sum(1)

ultrasensor_sum = np.zeros((ultra_fullamp.shape[0], 5))

for j in range(len(ultra_fullamp)):
    ultrasensor_sum[j][0] = ultrasensor_sum1[j]
    ultrasensor_sum[j][1] = ultrasensor_sum2[j]
    ultrasensor_sum[j][2] = ultrasensor_sum3[j]
    ultrasensor_sum[j][3] = ultrasensor_sum4[j]
    ultrasensor_sum[j][4] = ultrasensor_sum5[j]

ultrasensor_sum_binary = ultrasensor_sum.copy()

for i in range(len(ultrasensor_sum_binary)):
    for j in range(5):
        if ultrasensor_sum[i][j] > 0:
            ultrasensor_sum_binary[i][j] = 1
        else:
            ultrasensor_sum_binary[i][j] = 0

ultrasensor_sum_binary_label = np.insert(ultrasensor_sum_binary, 5, 3, axis=1)

np.savetxt('C:/Users/sohei/source/repos/pythonfft/pythonfft/1_mixer_3_test.csv', ultrasensor_sum_binary_label, delimiter=',', fmt='%d')

#np.savetxt('C:/Users/sohei/source/repos/pythonfft/pythonfft/27kHz.csv', fullamp, delimiter=',')


"""

#pandasのdataframe変換
"""
columns = ['25000Hz-27000Hz', '26000Hz-29000Hz', '29000Hz-31000Hz', '31000Hz-34000Hz', '39000Hz-41000Hz']
us = pd.DataFrame(data=ultrasensor_sum_binary, columns = columns)
us = us.assign(label=labelnumber) #ラベル
print(us)
"""



#print(amp_binary)




"""
with open(filename + '_binary.csv', 'w', newline='') as file_2:
    mywriter2 = csv.writer(file_2)
    mywriter2.writerow(arr_freqlist)
    mywriter2.writerows(amp_binary)


"""
#超音波帯のバイナリデータ
"""
with open(filename + '_ultra_binary.csv', 'w', newline='') as file_3:
    mywriter3 = csv.writer(file_3)
    mywriter3.writerow(ultra_freq)
    mywriter3.writerows(ultra_amp_binary)

"""
#5000Hzごとの振幅合計
"""
with open(filename + '_ultra_sum.csv', 'w', newline='') as file_4:
    mywriter4 = csv.writer(file_4)
    mywriter4.writerow(ultra_sum_freq)
    mywriter4.writerows(ultra_sum)

"""
#5000Hzごとの振幅バイナリデータ
"""
with open(filename + '_ultra_sum_binary.csv', 'w', newline='') as file_5:
    mywriter5 = csv.writer(file_5)
    mywriter5.writerow(ultra_sum_freq)
    mywriter5.writerows(ultra_sum_binary)
"""

#実際のセンサのピーク周波数帯の振幅合計
"""
with open(filename + '_ultrasensor_sum.csv', 'w', newline='') as file_6:
    mywriter6 = csv.writer(file_6)
    mywriter6.writerow(ultrasensor_sum_freq)
    mywriter6.writerows(ultrasensor_sum)

"""
#実際のセンサのピーク周波数帯のバイナリデータ（仮想データ）
"""
us.to_csv(filename + 'ultrasensor_sum_binary.csv')
"""
