import itertools
from mpl_settings import *
from consts import *
import matplotlib.pyplot as plt
import pandas as pd
from numpy.fft import fft, fftfreq
from datetime import datetime
import matplotlib.ticker as ticker
from functions import do_fft, phase_correction, unwrap, window


class Measurement:
    def __init__(self, data_td=None, meas_type=None, filepath=None, post_process_config=None):
        self.filepath = filepath
        self.meas_time = None
        self.meas_type = None
        self.sample_name = None
        self.position = [None, None]

        if post_process_config is None:
            from consts import post_process_config

        self.post_process_config = post_process_config
        self._data_fd, self._data_td = None, data_td
        self.pre_process_done = False

        self._set_metadata(meas_type)

    def __repr__(self):
        return str(self.filepath)

    def _set_metadata(self, meas_type=None):
        if meas_type is not None:
            self.meas_type = meas_type
            return

        # set time
        date_formats = [("%Y-%m-%dT%H-%M-%S.%f", 26), ("%Y-%m-%d_%H-%M-%S", 19)]
        for date_format in date_formats:
            try:
                length = date_format[1]
                self.meas_time = datetime.strptime(str(self.filepath.stem)[:length], date_format[0])
                break
            except ValueError:
                continue
        if self.meas_time is None:
            raise ValueError

        # set sample name
        try:
            dir_1above, dir_2above = self.filepath.parents[0], self.filepath.parents[1]
            if ("sam" in dir_1above.stem.lower()) or ("ref" in dir_1above.stem.lower()):
                self.sample_name = dir_2above.stem
            else:
                self.sample_name = dir_1above.stem
        except ValueError:
            self.sample_name = "N/A"

        # set measurement type
        if "ref" in str(self.filepath.stem).lower():
            self.meas_type = "ref"
        elif "sam" in str(self.filepath.stem).lower():
            self.meas_type = "sam"
        else:
            self.meas_type = "other"

        # set position
        try:
            str_splits = str(self.filepath).split("_")
            x = float(str_splits[-2].split(" mm")[0])
            y = float(str_splits[-1].split(" mm")[0])
            self.position = [x, y]
        except ValueError:
            self.position = [0, 0]

    def do_preprocess(self, force=False):
        if self.pre_process_done and not force:
            return

        if self.post_process_config["sub_offset"]:
            self._data_td[:, 1] -= np.mean(self._data_td[:10, 1])
        if self.post_process_config["en_windowing"]:
            self._data_td = window(self._data_td)
        if self.post_process_config["normalize"]:
            self._data_td[:, 1] /= np.max(self._data_td[:, 1])

        self.pre_process_done = True

    def get_data_td(self, get_raw=False):
        def read_file(file_path):
            try:
                return np.loadtxt(file_path)
            except ValueError:
                return np.loadtxt(file_path, delimiter=",")

        if get_raw:
            return read_file(self.filepath)

        if self._data_td is None:
            self._data_td = read_file(self.filepath)

        if self._data_td[0, 0] < 1:
            self._data_td[:, 0] *= 1E12

        self.do_preprocess()

        return self._data_td

    def get_data_fd(self, pos_freqs_only=True, reversed_time=True):
        if self._data_fd is not None:
            return self._data_fd

        data_td = self.get_data_td()
        t, y = data_td[:, 0], data_td[:, 1]

        if reversed_time:
            y = np.flip(y)

        dt = float(np.mean(np.diff(t)))
        freqs, data_fd = fftfreq(n=len(t), d=dt), fft(y)

        if pos_freqs_only:
            pos_slice = freqs >= 0
            self._data_fd = np.array([freqs[pos_slice], data_fd[pos_slice]]).T
        else:
            self._data_fd = np.array([freqs, data_fd]).T

        return self._data_fd


class MeasurementSet:
    all_measurements = None
    temp_log = None
    set_info = None
    avg_sam = None

    def __init__(self, data_dir=None, temp_log=None):
        self.data_dir = Path(data_dir)

        self.refs, self.sams = self._set_measurements()
        self.avg_sam = self._avg_sample()

        if temp_log is not None:
            self.temp_log = temp_log

        self._set_info()

    def _set_measurements(self):
        post_process = {"sub_offset": True, "en_windowing": False, "normalize": False}
        all_measurements = self._get_all_measurements(post_process=post_process)
        if len(all_measurements) == 0:
            exit(f"Exiting; no matching files found at {self.data_dir}")
        self.all_measurements = list(sorted(all_measurements, key=lambda meas: meas.meas_time))

        refs, sams = self._filter_measurements(self.all_measurements)

        return refs, sams

    def _get_all_measurements(self, post_process=None):
        measurements = []

        if self.data_dir is not None:
            glob = self.data_dir.glob("**/*.txt")
        else:
            glob = data_dir_ext.glob("**/*.txt")

        for file_path in glob:
            if file_path.is_file():
                try:
                    measurements.append(Measurement(filepath=file_path, post_process_config=post_process))
                except ValueError:
                    print(f"Skipping: {file_path} (Failed to parse time)")

        return measurements

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

        ref_cnt, sam_cnt = len(self.refs), len(self.sams)
        print(f"Number of reference and sample measurements in set: {ref_cnt}, {sam_cnt}")

        print("Time between first and last measurement: ",
              self.all_measurements[-1].meas_time - self.all_measurements[0].meas_time)

        self.set_info = {"dt": dt, "samples": samples, "ref_cnt": ref_cnt, "sam_cnt": sam_cnt}

    def _avg_sample(self):
        t = self.sams[0].get_data_td()[:, 0]

        y_arrays = []
        for measurement in self.sams:
            data_td = measurement.get_data_td()
            y_arrays.append(data_td[:, 1])

        return np.array([t, np.mean(y_arrays, axis=0)]).T

    def find_measurement(self, x, y):
        closest_sam, best_fit_val = None, np.inf
        for sam_meas in self.sams:
            val = abs(sam_meas.position[0] - x) + \
                  abs(sam_meas.position[1] - y)
            if val < best_fit_val:
                best_fit_val = val
                closest_sam = sam_meas

        return closest_sam

    def match_ref(self, measurement: Measurement, both_domains=False, normalize=False, sub_offset=False,
                  ret_meas=False):
        closest_ref, best_fit_val = None, np.inf
        for ref_meas in self.refs:
            val = np.abs((measurement.meas_time - ref_meas.meas_time).total_seconds())
            if val < best_fit_val:
                best_fit_val = val
                closest_ref = ref_meas
        print(f"Time between ref and sample: {(measurement.meas_time - closest_ref.meas_time).total_seconds()}")

        ref_td = closest_ref.get_data_td()

        if sub_offset:
            ref_td[:, 1] -= np.mean(ref_td[:, 1])

        if normalize:
            ref_td[:, 1] *= 1 / np.max(ref_td[:, 1])

        # ref_td[:, 0] -= ref_td[0, 0]

        if ret_meas:
            return closest_ref

        if both_domains:
            ref_fd = do_fft(ref_td)
            return ref_td, ref_fd
        else:
            return ref_td

    def system_stability(self, selected_freq_=None):
        def read_temp_file():
            df = pd.read_csv(self.temp_log)
            file_data = df.values

            times, temperatures = file_data[:, 0], file_data[:, 1]
            temperatures[temperatures > 30] /= 1e4
            temperatures[temperatures < 10] /= 1e-1

            times = (times - times[0]) / 3600

            return times, temperatures

        if selected_freq_ is None:
            selected_freq_ = self.freq_axis[len(self.freq_axis) // 2]

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

        fig, (ax1_amp, ax2_phase) = plt.subplots(nrows=2, ncols=1)

        ax1_amp.set_title("Single frequency reference amplitude")

        color = 'tab:blue'
        ax1_amp.set_xlabel("Measurement time (hour)")
        ax1_amp.set_ylabel("Amplitude (Arb. u.)")
        ax1_amp.plot(meas_times, ref_ampl_arr, label=f"Amplitude at {selected_freq_} THz")
        ax1_amp.tick_params(axis="y", labelcolor=color)
        ax1_amp.legend()

        ax2_phase.set_title("Single frequency reference phase")

        color = 'tab:blue'
        ax2_phase.set_xlabel("Measurement time (hour)")
        ax2_phase.set_ylabel("Phase (rad)")
        ax2_phase.plot(meas_times, ref_angle_arr, color=color, label=f"Phase at {selected_freq_} THz")
        ax2_phase.tick_params(axis="y", labelcolor=color)
        ax2_phase.legend()

        if self.temp_log is not None:
            temperature_time, temperature = read_temp_file()

            ax2 = ax1_amp.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:red'
            ax2.set_ylabel("Temperature (deg. C)", color=color)  # we already handled the x-label with ax1
            ax2.plot(temperature_time, temperature, color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped

            ax2 = ax2_phase.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:red'
            ax2.set_ylabel("Temperature (deg. C)", color=color)  # we already handled the x-label with ax1
            ax2.plot(temperature_time, temperature, color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped

    def calc_refractive_index(self, x, y, sam_id, en_plot=False):
        thickness = samples[str(sam_id)]
        measurement = self.find_measurement(x, y)

        ref_td, ref_fd = self.match_ref(measurement, both_domains=True)
        freqs = ref_fd[:, 0].real
        sam_fd = measurement.get_data_fd()

        phi_ref = phase_correction(ref_fd, en_plot=True)[:, 1]
        phi_sam = phase_correction(sam_fd, en_plot=True)[:, 1]

        delta_phi = phi_sam - phi_ref

        omega = 2 * pi * freqs
        n_a = 1 + c_thz * delta_phi / (omega * thickness)
        T = np.abs(sam_fd[:, 1] / ref_fd[:, 1])
        alpha = 1e4*(-2/thickness) * np.log(T * (n_a + 1)**2 / (4 * n_a))

        plot_range_ = plot_range_sub
        if sam_id == 5:
            plot_range_ = slice(25, 60)
        if sam_id == 4:
            plot_range_ = slice(25, 120)
        if sam_id == 3:
            plot_range_ = slice(25, 223)

        if en_plot:
            plt.figure("Analytical calculation")
            label = f"Sample {sam_id} ({x} mm, {y} mm, d={thickness} $\mu m$)"
            plt.subplot(1, 2, 1)
            plt.title("Analytical refractive index")
            plt.plot(freqs[plot_range_], n_a[plot_range_], label=label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Refractive index")

            plt.subplot(1, 2, 2)
            label = f"Sample {sam_id} ({x} mm, {y} mm, d={thickness} $\mu m$)"
            plt.title("Analytical absorption coefficient")
            plt.plot(freqs[plot_range_], alpha[plot_range_], label=label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Absorption coefficient $(cm^{-1})$")

        return array([freqs, n_a]).T, array([freqs, alpha]).T


class Image(MeasurementSet):
    """
    # loading times ....
    Todos:
        Part of measurement set ?
        - Refractive index + extinction coefficient image (single layer)
            - Add TeraLyzer thickness algo support
            - Birefringence?
            - Single points of above??
        - Compatibility with different dates possible?
        - Different measurement types
        - Ease of use (GUI?)
        - ...

    """

    _plotted_ref = False
    noise_floor = None
    time_axis = None
    cache_path = None
    sample_idx = None
    all_points = None
    name = ""

    def __init__(self, data_dir, sub_image=None, **kwargs):
        super().__init__(data_dir=data_dir)
        self.__dict__.update(kwargs)

        self.sub_image = sub_image  # second image (in case of substrate)

        self._set_image_info()
        self.image_data = self._image_cache()

    def _set_image_info(self):
        x_coords, y_coords = [], []
        for sam_measurement in self.sams:
            x_coords.append(sam_measurement.position[0])
            y_coords.append(sam_measurement.position[1])

        x_coords, y_coords = array(sorted(set(x_coords))), array(sorted(set(y_coords)))

        self.all_points = list(itertools.product(x_coords, y_coords))

        w, h = len(x_coords), len(y_coords)
        x_diff, y_diff = np.abs(np.diff(x_coords)), np.abs(np.diff(y_coords))
        dx = np.min(x_diff[np.nonzero(x_diff)])
        dy = np.min(y_diff[np.nonzero(y_diff)])

        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        self._empty_grid = np.zeros((w, h), dtype=complex)

        info = {"w": w, "h": h, "dx": dx, "dy": dy, "extent": extent}

        self.set_info.update(info)

    def _image_cache(self):
        """
        read all measurements into array and save as npy at location of first measurement
        """
        self.cache_path = Path(self.sams[0].filepath.parent / "cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)

        try:
            img_data = np.load(str(self.cache_path / "_raw_img_cache.npy"))
        except FileNotFoundError:
            w, h, samples = self.set_info["w"], self.set_info["h"], self.set_info["samples"]
            dx, dy = self.set_info["dx"], self.set_info["dy"]
            img_data = np.zeros((w, h, samples))
            min_x, max_x, min_y, max_y = self.set_info["extent"]

            for sam_measurement in self.sams:
                x_pos, y_pos = sam_measurement.position
                x_idx, y_idx = int((x_pos - min_x) / dx), int((y_pos - min_y) / dy)
                img_data[x_idx, y_idx] = sam_measurement.get_data_td(get_raw=True)[:, 1]

            np.save(str(self.cache_path / "_raw_img_cache.npy"), img_data)

        return img_data

    def _coords_to_idx(self, x, y):
        x_idx = int((x - self.set_info["extent"][0]) / self.set_info["dx"])
        y_idx = int((y - self.set_info["extent"][2]) / self.set_info["dy"])

        return x_idx, y_idx

    def _calc_power_grid(self, freq_range):
        def power(measurement):
            freq_slice = (freq_range[0] < self.freq_axis) * (self.freq_axis < freq_range[1])

            ref_td, ref_fd = self.match_ref(measurement, both_domains=True)

            sam_fd = measurement.get_data_fd()
            power_val_sam = np.sum(np.abs(sam_fd[freq_slice, 1])) / np.sum(freq_slice)
            power_val_ref = np.sum(np.abs(ref_fd[freq_slice, 1])) / np.sum(freq_slice)

            return (power_val_sam / power_val_ref) ** 2

        grid_vals = self._empty_grid.copy()

        for i, measurement in enumerate(self.sams):
            print(f"{round(100 * i / len(self.sams), 2)} % done. "
                  f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
            x_idx, y_idx = self._coords_to_idx(*measurement.position)
            val = power(measurement)
            grid_vals[x_idx, y_idx] = val

        return grid_vals

    def _calc_grid_vals(self, quantity="p2p", selected_freq=0.800):
        info = self.set_info

        if quantity.lower() == "power":
            if isinstance(selected_freq, tuple):
                grid_vals = self._calc_power_grid(freq_range=selected_freq)
            else:
                print("Selected frequency must be range given as tuple")
                grid_vals = self._empty_grid
        elif quantity == "p2p":
            grid_vals = np.max(self.image_data, axis=2) - np.min(self.image_data, axis=2)
        elif quantity.lower() == "ref_amp":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                amp_, _ = self._ref_interpolation(measurement, selected_freq_=selected_freq,
                                                  ret_cart=False)
                grid_vals[x_idx, y_idx] = amp_
        elif quantity == "ref_phi":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                _, phi_ = self._ref_interpolation(measurement, selected_freq_=selected_freq,
                                                  ret_cart=False)
                grid_vals[x_idx, y_idx] = phi_
        else:
            # grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)
            grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)

        return grid_vals.real

    def plot_image(self, selected_freq=None, quantity="p2p", img_extent=None):
        if quantity.lower() == "p2p":
            label = ""
        elif quantity.lower() == "ref_amp":
            label = " Interpolated ref. amp. at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity == "ref_phi":
            label = " interpolated at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity.lower() == "power":
            label = f"({selected_freq[0]}-{selected_freq[1]}) THz"
        else:
            label = ""

        grid_vals = self._calc_grid_vals(quantity=quantity, selected_freq=selected_freq)

        info = self.set_info
        if img_extent is None:
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        grid_vals = grid_vals[w0:w1, h0:h1]

        fig = plt.figure(f"{self.name}")
        ax = fig.add_subplot(111)
        ax.set_title(f"{self.name}")
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.set_info["extent"]

        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                        origin="lower",
                        cmap=plt.get_cmap('jet'),
                        extent=img_extent)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        def fmt(x, val):
            a, b = '{:.2e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)

        cbar = fig.colorbar(img, format=ticker.FuncFormatter(fmt))
        cbar.set_label(f"{quantity}" + label, rotation=270, labelpad=30)

    def get_point(self, x, y, normalize=False, sub_offset=False, both=False, add_plot=False):
        dx, dy, dt = self.set_info["dx"], self.set_info["dy"], self.set_info["dt"]

        x_idx, y_idx = self._coords_to_idx(x, y)
        y_ = self.image_data[x_idx, y_idx]

        if sub_offset:
            y_ -= np.mean(y_)

        if normalize:
            y_ *= 1 / np.max(y_)

        t = np.arange(0, len(y_)) * dt
        y_td = np.array([t, y_]).T

        if add_plot:
            self.plot_point(x, y, y_td)

        if not both:
            return y_td
        else:
            return y_td, do_fft(y_td)

    def plot_point(self, x, y, sam_id=None, sam_td=None, ref_td=None, sub_noise_floor=False, td_scale=1):
        if (sam_td is None) and (ref_td is None):
            measurement = self.find_measurement(x, y)
            print(measurement)
            sam_td = measurement.get_data_td()
            ref_td = self.match_ref(measurement, sub_offset=False)

            # sam_td = window(sam_td, win_len=25, shift=0, en_plot=False, slope=0.05)
            # ref_td = window(ref_td, win_len=25, shift=0, en_plot=False, slope=0.05)

            ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

            # sam_td, sam_fd = phase_correction(sam_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)
            # ref_td, ref_fd = phase_correction(ref_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)
        else:
            ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

        ref_td[:, 1] -= np.mean(ref_td[:, 1])
        sam_td[:, 1] -= np.mean(sam_td[:, 1])

        phi_ref, phi_sam = unwrap(ref_fd), unwrap(sam_fd)

        noise_floor = np.mean(20 * np.log10(np.abs(ref_fd[ref_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        if not self._plotted_ref:
            plt.figure("Spectrum")
            plt.plot(ref_fd[plot_range1, 0], 20 * np.log10(np.abs(ref_fd[plot_range1, 1])) - noise_floor,
                     label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

            plt.figure("Phase")
            plt.plot(ref_fd[plot_range1, 0], phi_ref[plot_range1, 1], label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase (rad)")

            plt.figure("Time domain")
            plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (arb. u.)")

            self._plotted_ref = True

        label = f"(x={x} mm, y={y} mm)"
        if sam_id is not None:
            label = f"Sample {sam_id} " + label
            label = label.replace(" mm)", f" mm, d={samples[str(sam_id)]} $\mu m$)")

        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        plt.figure("Spectrum")
        plt.plot(sam_fd[plot_range1, 0], 20 * np.log10(np.abs(sam_fd[plot_range1, 1])) - noise_floor, label=label)

        plt.figure("Phase")
        plt.plot(sam_fd[plot_range1, 0], phi_sam[plot_range1, 1], label=label)

        plt.figure("Time domain")
        plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=label)

    def _ref_interpolation(self, sam_meas, selected_freq_=0.800, ret_cart=False):
        sam_meas_time = sam_meas.meas_time

        nearest_ref_idx, smallest_time_diff, time_diff = None, np.inf, 0
        for ref_idx in range(len(self.refs)):
            time_diff = (self.refs[ref_idx].meas_time - sam_meas_time).total_seconds()
            if abs(time_diff) < abs(smallest_time_diff):
                nearest_ref_idx = ref_idx
                smallest_time_diff = time_diff

        t0 = self.refs[0].meas_time
        if smallest_time_diff <= 0:
            # sample was measured after reference
            ref_before = self.refs[nearest_ref_idx]
            ref_after = self.refs[nearest_ref_idx + 1]
        else:
            ref_before = self.refs[nearest_ref_idx - 1]
            ref_after = self.refs[nearest_ref_idx]

        t = [(ref_before.meas_time - t0).total_seconds(), (ref_after.meas_time - t0).total_seconds()]
        ref_before_td, ref_after_td = ref_before.get_data_td(), ref_after.get_data_td()

        # ref_before_td = window(ref_before_td, win_len=12, shift=0, en_plot=False, slope=0.05)
        # ref_after_td = window(ref_after_td, win_len=12, shift=0, en_plot=False, slope=0.05)

        ref_before_fd, ref_after_fd = do_fft(ref_before_td), do_fft(ref_after_td)

        # ref_before_fd = phase_correction(ref_before_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)
        # ref_after_fd = phase_correction(ref_after_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)

        # if isinstance(selected_freq_, tuple):

        # else:
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))
        y_amp = [np.sum(np.abs(ref_before_fd[f_idx, 1])) / 1,
                 np.sum(np.abs(ref_after_fd[f_idx, 1])) / 1]
        y_phi = [np.angle(ref_before_fd[f_idx, 1]), np.angle(ref_after_fd[f_idx, 1])]

        amp_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_amp)
        phi_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_phi)

        if ret_cart:
            return amp_interpol * np.exp(1j * phi_interpol)
        else:
            return amp_interpol, phi_interpol


if __name__ == '__main__':
    data_path = data_dir_ext / "Image0"
    image = Image(data_dir=data_path)

    image.plot_image(img_extent=[-10, 75, -3, 27], quantity="p2p")

    # sub_image.system_stability(selected_freq_=0.800)
    # film_image.system_stability(selected_freq_=1.200)

    """
    1	10, -1
    2	7, 19 
    3	33, 20
    4	57, 20
    5	33, -1
    """

    image.plot_point(10, -1, sam_id=1)
    image.plot_point(7, 19, sam_id=2)
    image.plot_point(33, 19, sam_id=3)
    image.plot_point(57, 19, sam_id=4)
    image.plot_point(33, 1, sam_id=5)

    image.calc_refractive_index(10, -1, sam_id=1, en_plot=True)
    image.calc_refractive_index(7, 19, sam_id=2, en_plot=True)
    image.calc_refractive_index(33, 19, sam_id=3, en_plot=True)
    image.calc_refractive_index(57, 19, sam_id=4, en_plot=True)
    image.calc_refractive_index(33, -1, sam_id=5, en_plot=True)

    for fig_label in plt.get_figlabels():
        if "Sample" not in fig_label:
            pass
        plt.figure(fig_label)
        plt.legend()

    plt.show()
