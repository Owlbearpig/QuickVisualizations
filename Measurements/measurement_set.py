import numpy as np

from measurements import get_all_measurements
from functions import do_fft
from consts import *
import matplotlib.pyplot as plt
import pandas as pd


class MeasurementSet:
    all_measurements = None

    def __init__(self, data_path):
        self.data_path = data_path

        self.refs, self.sams = self._set_measurements()
        self._set_info()

    def _set_measurements(self):
        post_process = {"sub_offset": False, "en_windowing": False, "normalize": True}
        all_measurements = get_all_measurements(data_dir_=self.data_path, post_process=post_process)
        self.all_measurements = all_measurements

        refs, sams = self._filter_measurements(all_measurements)
        refs = tuple(sorted(refs, key=lambda meas: meas.meas_time))
        sams = tuple(sorted(sams, key=lambda meas: meas.meas_time))

        return refs, sams

    @staticmethod
    def _filter_measurements(measurements):
        refs, sams = [], []
        for measurement in measurements:
            if measurement.meas_type == "ref":
                refs.append(measurement)
            else:
                sams.append(measurement)

        return refs, sams

    def _set_info(self):
        example_measurement = self.all_measurements[0]
        data_td = example_measurement.get_data_td()
        samples = int(data_td.shape[0])
        self.time_axis = data_td[:, 0].real

        sample_data_fd = example_measurement.get_data_fd()
        self.freq_axis = sample_data_fd[:, 0].real

        dt = np.mean(np.diff(self.time_axis))

    def system_stability(self, selected_freq_=0.800):
        def read_temp_file(file_path):
            df = pd.read_csv(file_path)
            file_data = df.values

            times, temperatures = file_data[:, 0], file_data[:, 1]
            temperatures[temperatures > 30] /= 1e4
            temperatures[temperatures < 10] /= 1e-1

            times = (times - times[0]) / 3600

            return times, temperatures

        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))

        ref_ampl_arr, ref_angle_arr = [], []

        t0 = self.refs[0].meas_time
        meas_times = [(ref.meas_time - t0).total_seconds() / 3600 for ref in self.refs]
        for i, ref in enumerate(self.refs):
            ref_td = ref.get_data_td()
            ref_fd = do_fft(ref_td)

            ref_ampl_arr.append(np.sum(np.abs(ref_fd[f_idx, 1])) / 1)
            phi = np.angle(ref_fd[f_idx, 1])
            if i and (abs(ref_angle_arr[-1] - phi) > pi):
                phi -= 2 * pi
            ref_angle_arr.append(phi)

        temperature_time, temperature = read_temp_file("E:\measurementdata\Misc\Klimaanlagenbeeinflussung\Temperatursensor.csv")

        fig, ax1 = plt.subplots()

        ax1.set_title("Single frequency reference amplitude")

        color = 'tab:blue'
        ax1.set_xlabel("Measurement time (hour)")
        ax1.set_ylabel("Amplitude (Arb. u.)")
        ax1.plot(meas_times, ref_ampl_arr, label=f"Amplitude at {selected_freq_} THz")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.legend()

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel("Temperature (deg. C)", color=color)  # we already handled the x-label with ax1
        ax2.plot(temperature_time, temperature, color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        fig, ax1 = plt.subplots()

        ax1.set_title("Single frequency reference phase")

        color = 'tab:blue'
        ax1.set_xlabel("Measurement time (hour)")
        ax1.set_ylabel("Phase (rad)")
        ax1.plot(meas_times, ref_angle_arr, color=color, label=f"Phase at {selected_freq_} THz")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.legend()

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel("Temperature (deg. C)", color=color)  # we already handled the x-label with ax1
        ax2.plot(temperature_time, temperature, color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped


if __name__ == '__main__':
    measurement_set = MeasurementSet(data_dir)
    measurement_set.system_stability(selected_freq_=0.140)

    for fig_label in plt.get_figlabels():
        if "Sample" not in fig_label:
            continue
        plt.figure(fig_label)
        plt.legend()

    plt.show()
