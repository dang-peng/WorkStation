import matplotlib.pyplot as plt
import numpy as np

time_of_view = 1  # s.
# analog_time = np.linspace(0, time_of_view, 10e5)  # s.
analog_time = np.linspace(0, time_of_view, num = 100000)  # s.

sampling_rate = 60  # Hz
sampling_period = 1 / sampling_rate  # s
sample_number = int(time_of_view / sampling_period)
sampling_time = np.linspace(0, time_of_view, sample_number)

carrier_frequency = 9
amplitude = 1
phase = 0

quantizing_bits = 4
quantizing_levels = 2 ** quantizing_bits / 2
quantizing_step = 1 / quantizing_levels


def analog_signal(time_point):
    return amplitude * np.cos(2 * np.pi * carrier_frequency * time_point + phase)


sampling_signal = analog_signal(sampling_time)
quantization_signal = np.round(sampling_signal / quantizing_step) * quantizing_step

fig = plt.figure()
plt.plot(analog_time, analog_signal(analog_time))
baseline = plt.stem(sampling_time, quantization_signal, linefmt = 'r-', markerfmt = 'rs',
                    basefmt = 'r-')
# plt.setp(stemlines, 'linewidth', 2)  # 设置线宽度
# plt.setp(markerline, 'markersize', 8)  # 设置标记大小

plt.title("Analog to digital signal conversion")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.show()
