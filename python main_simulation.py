import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List


# =========================================================
# بخش A) توابع پایه‌ی زمان/گیت (برای هر دو مدل)
# =========================================================

def update_V_coh(fiber_length_km: float, V_coh_init: float, decay_rate: float = 0.0006) -> float:
    """
    به‌روزرسانی V_coh بر اساس طول فیبر و نرخ کاهش.
    """
    return V_coh_init * np.exp(-decay_rate * fiber_length_km)


def fwhm_to_sigma(fwhm: float) -> float:
    """FWHM -> sigma برای گاوسی (همان تابع شما)"""
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gauss_cdf(t: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if t >= mu else 0.0
    return 0.5 * (1.0 + math.erf((t - mu) / (sigma * math.sqrt(2))))


def fraction_overlap(pulse_center: float, window_center: float, window_width: float, sigma: float) -> float:
    """کسری از پالس گاوسی که داخل گیت می‌افتد (واحدها: هرچی بدهید همان)"""
    t_start = window_center - window_width / 2.0
    t_end = window_center + window_width / 2.0
    return max(0.0, min(1.0, gauss_cdf(t_end, pulse_center, sigma) - gauss_cdf(t_start, pulse_center, sigma)))


def gaussian_gate_fraction(t_center: float, t_gate_center: float, sigma_total: float, gate_width: float) -> float:
    """
    نسخه‌ی عمومی همان تابع CHSH شما (با erf)،
    اینجا واحدها آزاد است: ps یا s فرقی ندارد اگر همه یکسان باشند.
    """
    if sigma_total <= 0 or gate_width <= 0:
        return 0.0
    z_hi = (t_gate_center + gate_width / 2.0 - t_center) / (math.sqrt(2.0) * sigma_total)
    z_lo = (t_gate_center - gate_width / 2.0 - t_center) / (math.sqrt(2.0) * sigma_total)
    return 0.5 * (math.erf(z_hi) - math.erf(z_lo))


# =========================================================
# بخش جدید: تابع محاسبه SKR
# =========================================================

def compute_skr(S: float, Q: float, raw_rate: float) -> float:
    """
    محاسبه نرخ کلید امن (Secure Key Rate) با فرمول داده شده.
    
    فرمول: SKR = [1 - log₂(1 + √(2 - S²/4)) - 1.1 * h(Q)] * Raw Rate
    
    که در آن:
    - S: پارامتر CHSH (از شاخه نظارت)
    - Q: QBER (از شاخه دیتا)
    - raw_rate: نرخ خام به ازای هر پالس ارسالی (Overall) - استفاده از RawRate_total_sent
    - h(Q) = -Q*log₂(Q) - (1-Q)*log₂(1-Q)
    
    توجه: برای Q = 0 یا 1، h(Q) = 0 تعریف می‌کنیم.
    """
    # محاسبه h(Q) - آنتروپی باینری
    if Q <= 0 or Q >= 1:
        h_Q = 0.0
    else:
        h_Q = -Q * math.log2(Q) - (1 - Q) * math.log2(1 - Q)
    
    # محاسبه ترم داخل فرمول
    if S <= 0:
        skr_term = 0.0
    else:
        skr_term = 1 - math.log2(1 + math.sqrt(2 - (S**2) / 4)) - 1.1 * h_Q
    
    # SKR نهایی (اگر ترم منفی باشد، SKR = 0 در نظر می‌گیریم)
    if skr_term <= 0:
        return 0.0
    
    return skr_term * raw_rate


# =========================================================
# بخش B) منبع حالت آلیس (مشترک برای هر دو شاخه)
# =========================================================

class AliceStateSource:
    """
    آلیس دامنه‌های پیچیده Early/Late را تولید می‌کند.
    - برای دیتالاین فقط |aE|^2 و |aL|^2 مهم است.
    - برای CHSH دامنه‌ی پیچیده (فاز/علامت) مهم است.

    نگاشت پیشنهادی (برای سازگار شدن با کدهای شما):
    - '0'  : Z0  -> (alpha, 0)     
    - '1'  : Z1  -> (0, alpha)
    - 'd'  : DEC_PP -> (alpha/sqrt2, +alpha/sqrt2)
    - 'f'  : DEC_MP -> (-alpha/sqrt2, +alpha/sqrt2)
      (در دیتالاین d و f از نظر شدت یکی هستند، پس خروجی دیتالاین تغییر نمی‌کند؛
       ولی برای CHSH لازم است f علامت مخالف داشته باشد.)
    """

    def __init__(self, amplitude_a: float,
                 p0: float = 0.45, p1: float = 0.45, pd: float = 0.05, pf: float = 0.05):
        self.alpha = float(amplitude_a)
        self.probs = {'0': p0, '1': p1, 'd': pd, 'f': pf}
        self.states = self._build_states()

    def _build_states(self) -> Dict[str, Dict[str, complex]]:
        a = self.alpha
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return {
            '0': {'aE': complex(a),            'aL': complex(0.0)},
            '1': {'aE': complex(0.0),          'aL': complex(a)},
            'd': {'aE': complex(a * inv_sqrt2),'aL': complex(a * inv_sqrt2)},     # DEC_PP
            'f': {'aE': complex(-a * inv_sqrt2),'aL': complex(a * inv_sqrt2)}     # DEC_MP
        }

    def list_states(self):
        return list(self.states.keys())

    def get(self, label: str) -> Tuple[complex, complex, float]:
        st = self.states[label]
        return st['aE'], st['aL'], self.probs[label]


# =========================================================
# بخش C) شاخه Data-line (QBER + Raw Rate) — اصلاح فقط مخرج به سیگنال‌ها
# =========================================================

def calculate_raw_photon_probabilities(mu_sent_E: float, mu_sent_L: float, params: dict) -> Tuple[float, float]:
    """
    دقیقا منطق شما، با همان پارامترها (واحد ثانیه).
    خروجی: (p_phot_E, p_phot_L)
    """
    L = params['fiber_length_km']
    loss = params['loss_dB_per_km']
    eta_det = params['detector_efficiency']
    D = params['chromatic_dispersion']
    d_lambda = params['spectral_width_nm']

    sigma_pulse = fwhm_to_sigma(params['pulse_fwhm_s'])
    sigma_jitter = fwhm_to_sigma(params['jitter_fwhm_s'])

    gate_w = params['gate_width_s']
    tau = params['bin_separation_s']

    # D[dps/(nm*km)] * d_lambda[nm] * L[km] -> ps  => تبدیل به ثانیه
    sig_dispersion_val_s = (D * d_lambda * L) * 1e-12
    sigma_dispersion = fwhm_to_sigma(sig_dispersion_val_s)
    sigma_total = math.sqrt(sigma_pulse**2 + sigma_jitter**2 + sigma_dispersion**2)

    channel_transmittance = 10 ** (-(loss * L) / 10.0)
    eta_total = channel_transmittance * eta_det

    # overlap ها
    f_EE = fraction_overlap(0.0, 0.0, gate_w, sigma_total)
    f_EL = fraction_overlap(0.0, tau, gate_w, sigma_total)
    f_LE = fraction_overlap(tau, 0.0, gate_w, sigma_total)
    f_LL = fraction_overlap(tau, tau, gate_w, sigma_total)

    mu_in_Early = eta_total * (mu_sent_E * f_EE + mu_sent_L * f_LE)
    mu_in_Late  = eta_total * (mu_sent_L * f_LL + mu_sent_E * f_EL)

    p_phot_E = 1.0 - math.exp(-mu_in_Early)
    p_phot_L = 1.0 - math.exp(-mu_in_Late)
    return p_phot_E, p_phot_L


class DataLineQKD:
    """
    شاخه دیتا:
    - از منبع حالت آلیس (aE, aL) استفاده می‌کند ولی فقط شدت‌ها را می‌گیرد.
    - همان منطق QBER/RawRate کد شما + afterpulse mean-field.
    """

    def __init__(self, params: dict, alice: AliceStateSource):
        self.params = params
        self.alice = alice

    def calculate_average_system_load(self) -> float:
        avg_click_prob = 0.0
        p_dark = self.params['dark_count_prob_per_gate']

        for label in self.alice.list_states():
            aE, aL, w = self.alice.get(label)
            mu_E = abs(aE)**2
            mu_L = abs(aL)**2

            p_ph_E, p_ph_L = calculate_raw_photon_probabilities(mu_E, mu_L, self.params)
            p_c_E = 1.0 - (1.0 - p_ph_E) * (1.0 - p_dark)
            p_c_L = 1.0 - (1.0 - p_ph_L) * (1.0 - p_dark)
            avg_click_prob += w * (p_c_E + p_c_L)

        return avg_click_prob

    def run(self, verbose: bool = True) -> Tuple[float, float]:
        """
        خروجی:
        - qber (شرطی روی شاخه دیتا و حالت‌های سیگنال)
        - raw_rate_per_signal_pulse_in_data (به ازای هر پالس سیگنال که وارد شاخه دیتا شده)
        """
        avg_load = self.calculate_average_system_load()
        p_ap_prob = self.params.get('after_pulse_prob', 0.0)

        p_ap_noise = avg_load * p_ap_prob / 2.0

        if verbose:
            print(f"\n--- Data-line Analysis (L={self.params['fiber_length_km']} km) ---")
            print(f"Average Detector Load:       {avg_load:.4e}")
            print(f"Effective Afterpulse Noise:  {p_ap_noise:.4e}")
            print("-" * 78)
            print(f"{'State':<6} | {'P(Click E)':<12} | {'P(Click L)':<12} | {'P(Error)':<12}")

        prob_valid_signal = 0.0
        prob_error_signal = 0.0
        prob_signal = 0.0  # مجموع وزن حالت‌های سیگنال (w0 + w1)
        p_dark = self.params['dark_count_prob_per_gate']

        for label in self.alice.list_states():
            aE, aL, w = self.alice.get(label)
            mu_E = abs(aE)**2
            mu_L = abs(aL)**2

            p_ph_E, p_ph_L = calculate_raw_photon_probabilities(mu_E, mu_L, self.params)

            p_no_click_E = (1.0 - p_ph_E) * (1.0 - p_dark) * (1.0 - p_ap_noise)
            p_no_click_L = (1.0 - p_ph_L) * (1.0 - p_dark) * (1.0 - p_ap_noise)
            p_click_E = 1.0 - p_no_click_E
            p_click_L = 1.0 - p_no_click_L

            p_err = 0.0
            p_correct = 0.0

            # سیگنال‌ها فقط '0' و '1' هستند
            if label == '0':
                p_correct = p_click_E * (1.0 - p_click_L)
                p_err     = p_click_L * (1.0 - p_click_E)
            elif label == '1':
                p_correct = p_click_L * (1.0 - p_click_E)
                p_err     = p_click_E * (1.0 - p_click_L)

            if verbose:
                print(f"{label:<6} | {p_click_E:.4e}   | {p_click_L:.4e}   | {p_err:.4e}")

            if label in ['0', '1']:
                prob_signal += w
                prob_valid_signal += w * (p_correct + p_err)
                prob_error_signal += w * p_err

        qber = (prob_error_signal / prob_valid_signal)+0.01 if prob_valid_signal > 0 else 0.5

        # نرخ خام فقط بر اساس پالس‌های سیگنال (اصلاح مخرج)
        raw_rate_per_signal_pulse = (prob_valid_signal / prob_signal) if prob_signal > 0 else 0.0

        if verbose:
            print("-" * 78)
            print(f"P(signal) = P('0')+P('1') = {prob_signal:.4f}")
            print(f"Raw Rate in DATA (per signal pulse):  {raw_rate_per_signal_pulse:.4e}")
            print(f"QBER in DATA (conditioned):           {qber * 100:.4f} %")

        return qber, raw_rate_per_signal_pulse *0.05


# =========================================================
# بخش D) شاخه Monitoring (CHSH / S) — منطق کد دوم
# =========================================================

def click_probability(mu_eff: float, p_dark: float) -> float:
    return 1.0 - math.exp(-mu_eff) * (1.0 - p_dark)


def compute_middle_gate_intensities(E_short_early, E_long_early,
                                   E_short_late, E_long_late,
                                   phi_B, amp_imb, phi_mis, V_coh,
                                   c11, c12, c21, c22,
                                   F_mid, F_E2M, F_L2M, mu_scale):
    """
    همان تابع شما (بدون تغییر منطقی)
    """
    phase_total = phi_B + phi_mis
    amp_factor = 1.0 + amp_imb

    E_L_input = (E_long_early * math.sqrt(F_mid) +
                 E_long_late * math.sqrt(F_L2M)) * amp_factor * np.exp(1j * phase_total)

    E_S_input = (E_short_late * math.sqrt(F_mid) +
                 E_short_early * math.sqrt(F_E2M))

    A1_L = c11 * E_L_input
    A1_S = c12 * E_S_input
    A2_L = c21 * E_L_input
    A2_S = c22 * E_S_input

    I1_nonint = abs(A1_L)**2 + abs(A1_S)**2
    I2_nonint = abs(A2_L)**2 + abs(A2_S)**2

    I1_int = 2.0 * np.real(A1_L * np.conj(A1_S))
    I2_int = 2.0 * np.real(A2_L * np.conj(A2_S))

    I1 = mu_scale * (I1_nonint + V_coh * I1_int)
    I2 = mu_scale * (I2_nonint + V_coh * I2_int)

    return I1, I2


class MonitoringCHSH:
    """
    شاخه نظارت:
    - از همان منبع حالت آلیس استفاده می‌کند ولی اینجا دامنه‌ی پیچیده مهم است.
    - حالت‌های لازم برای CHSH:
      A0: Z0 vs Z1  (یعنی '0' و '1')
      A1: DEC_PP vs DEC_MP (یعنی 'd' و 'f')
    """

    def __init__(self, params: dict, alice: AliceStateSource):
        self.params = params
        self.alice = alice

    def analytical_CHSH(self) -> Dict[str, float]:
        p = self.params

        # کانال
        eta_channel = 10 ** (-(p['loss_dB_per_km'] * p['fiber_length_km']) / 10.0)
        eta_det = p.get('eta_det_main', 0.2)
        mu_scale = eta_channel * eta_det

        # زمان‌ها (واحد: ps در مدل CHSH)
        sigma_pulse_ps = p.get('pulse_sigma_ps', None)
        sigma_jitter_ps = p.get('jitter_sigma_ps', None)

        # اگر کاربر فقط پارامترهای ثانیه‌ای داده باشد، از آنها به ps تبدیل می‌کنیم
        # (این بخش اختیاری است ولی خیلی کمک می‌کند کد یکپارچه شود)
        if sigma_pulse_ps is None:
            # از pulse_fwhm_s -> sigma_s -> ps
            sigma_pulse_ps = fwhm_to_sigma(p['pulse_fwhm_s']) * 1e12
        if sigma_jitter_ps is None:
            sigma_jitter_ps = fwhm_to_sigma(p['jitter_fwhm_s']) * 1e12

        D_ps_nm_km = p.get('dispersion_D_ps_nm_km', p.get('chromatic_dispersion', 17.0))
        d_lambda_nm = p.get('spectral_width_nm', 0.1)
        L_km = p['fiber_length_km']
        sigma_disp_ps = abs(D_ps_nm_km) * d_lambda_nm * L_km
        sigma_total_ps = math.sqrt(sigma_pulse_ps**2 + sigma_jitter_ps**2 + sigma_disp_ps**2)

        gate_w_ps = p.get('gate_width_ps', None)
        dt_ps = p.get('bin_separation_ps', None)

        # اگر کاربر فقط ثانیه‌ای داده باشد، از آنها به ps تبدیل می‌کنیم
        if gate_w_ps is None:
            gate_w_ps = p['gate_width_s'] * 1e12
        if dt_ps is None:
            dt_ps = p['bin_separation_s'] * 1e12

        # آشکارساز/پس‌زمینه
        p_dark = p.get('p_dark_main', p.get('dark_count_prob_per_gate', 1e-6))
        mu_bg = p.get('mu_bg_main', 0.0)

        # Misalignment / visibility
        phi_mis = p.get('phi_mis', 0.0)
        amp_imb = p.get('amp_imbalance', 0.0)
        #V_coh = p.get('V_coh', 1.0)
        V_coh = update_V_coh(fiber_length_km=L_km, V_coh_init=0.99)

        # BS1
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        BS1 = np.array([[inv_sqrt2, 1j * inv_sqrt2],
                        [1j * inv_sqrt2, inv_sqrt2]], dtype=complex)

        # BS2
        R2 = p.get('BS2_R', 0.85)
        T2 = 1.0 - R2
        c11 = math.sqrt(R2)
        c12 = 1j * math.sqrt(T2)
        c21 = 1j * math.sqrt(T2)
        c22 = math.sqrt(R2)

        # مراکز زمانی E/M/L (برای مدل نشت در گیت middle)
        tE = 0.0
        tM = dt_ps
        tL = 2.0 * dt_ps

        # fraction ها
        F_mid = gaussian_gate_fraction(tM, tM, sigma_total_ps, gate_w_ps)
        F_E2M = gaussian_gate_fraction(tE, tM, sigma_total_ps, gate_w_ps)
        F_L2M = gaussian_gate_fraction(tL, tM, sigma_total_ps, gate_w_ps)

        # حالت‌ها از آلیس (همان چهار state)
        mapping = {
            "Z0": "0",
            "Z1": "1",
            "DEC_PP": "d",
            "DEC_MP": "f",
        }

        results = { "B0": {}, "B1": {} }
        for st_label, alice_label in mapping.items():
            aE, aL, _w = self.alice.get(alice_label)

            # ورودی‌های early/late به BS1
            vec_early = np.array([aE, 0.0], dtype=complex)
            vec_late = np.array([aL, 0.0], dtype=complex)

            arms_early = BS1 @ vec_early
            arms_late = BS1 @ vec_late

            E_short_early = arms_early[0]
            E_long_early  = arms_early[1]
            E_short_late  = arms_late[0]
            E_long_late   = arms_late[1]

            # B0
            I1_B0, I2_B0 = compute_middle_gate_intensities(
                E_short_early, E_long_early, E_short_late, E_long_late,
                phi_B=0.0, amp_imb=amp_imb, phi_mis=phi_mis, V_coh=V_coh,
                c11=c11, c12=c12, c21=c21, c22=c22,
                F_mid=F_mid, F_E2M=F_E2M, F_L2M=F_L2M, mu_scale=mu_scale
            )

            # B1
            I1_B1, I2_B1 = compute_middle_gate_intensities(
                E_short_early, E_long_early, E_short_late, E_long_late,
                phi_B=math.pi, amp_imb=amp_imb, phi_mis=phi_mis, V_coh=V_coh,
                c11=c11, c12=c12, c21=c21, c22=c22,
                F_mid=F_mid, F_E2M=F_E2M, F_L2M=F_L2M, mu_scale=mu_scale
            )

            # background
            I1_tot_B0 = I1_B0 + mu_bg
            I2_tot_B0 = I2_B0 + mu_bg
            I1_tot_B1 = I1_B1 + mu_bg
            I2_tot_B1 = I2_B1 + mu_bg

            results["B0"][st_label] = {
                "m1": click_probability(I1_tot_B0, p_dark),
                "m2": click_probability(I2_tot_B0, p_dark)
            }
            results["B1"][st_label] = {
                "m1": click_probability(I1_tot_B1, p_dark),
                "m2": click_probability(I2_tot_B1, p_dark)
            }

        def get_E(state_plus: str, state_minus: str, B_setting: str) -> float:
            m1_p = results[B_setting][state_plus]["m1"]
            m2_p = results[B_setting][state_plus]["m2"]
            m1_m = results[B_setting][state_minus]["m1"]
            m2_m = results[B_setting][state_minus]["m2"]
            num = (m1_p - m2_p) - (m1_m - m2_m)
            den = (m1_p + m2_p) + (m1_m + m2_m)
            return (num / den) if den > 0 else 0.0

        E_A0_B0 = get_E("Z0", "Z1", "B0")
        E_A0_B1 = get_E("Z0", "Z1", "B1")
        E_A1_B0 = get_E("DEC_PP", "DEC_MP", "B0")
        E_A1_B1 = get_E("DEC_PP", "DEC_MP", "B1")

        S_value = E_A0_B0 + E_A0_B1 + E_A1_B0 - E_A1_B1

        return {
            "E_A0_B0": float(E_A0_B0),
            "E_A0_B1": float(E_A0_B1),
            "E_A1_B0": float(E_A1_B0),
            "E_A1_B1": float(E_A1_B1),
            "S_value": float(S_value),
            "F_mid": float(F_mid),
            "F_E2M": float(F_E2M),
            "F_L2M": float(F_L2M),
            "sigma_total_ps": float(sigma_total_ps)
        }


# =========================================================
# بخش E) ادغام نهایی: Bob Router با احتمال 90/10 + SKR
# =========================================================

class HybridBobRouter:
    """
    هسته‌ی ادغام:
    - آلیس مشترک
    - باب با احتمال p_data شاخه دیتا و با احتمال p_mon شاخه نظارت

    خروجی‌ها:
    - QBER_data: فقط برای دیتالاین (شرطی)
    - RawRate_in_data: نرخ خام به ازای پالس‌های سیگنال در دیتالاین (شرطی)
    - RawRate_total_sent: نرخ خام به ازای هر پالس ارسالی (p_data × P(signal) × RawRate_in_data)
    - S_monitor: پارامتر S برای شاخه نظارت (شرطی)
    - SKR: نرخ کلید امن محاسبه شده با فرمول داده شده
    """

    def __init__(self, params: dict, p_data: float = 0.9):
        self.params = params
        self.p_data = float(p_data)
        self.p_mon = 1.0 - self.p_data

        self.alice = AliceStateSource(
            amplitude_a=params['amplitude_a'],
            p0=params.get('prob_0', 0.45),
            p1=params.get('prob_1', 0.45),
            pd=params.get('prob_d', 0.05),
            pf=params.get('prob_f', 0.05),
        )

    def run_once(self, verbose: bool = True) -> Dict[str, Any]:
        data_sim = DataLineQKD(self.params, self.alice)
        mon_sim = MonitoringCHSH(self.params, self.alice)

        qber_data, raw_in_data = data_sim.run(verbose=verbose)
        mon_out = mon_sim.analytical_CHSH()
        
        # احتمال سیگنال‌ها در آلیس
        p_signal = self.alice.probs['0'] + self.alice.probs['1']
        
        # محاسبه RawRate_total_sent و SKR
        raw_total_sent = self.p_data * p_signal * raw_in_data
        skr = compute_skr(mon_out["S_value"], qber_data, raw_total_sent)

        out = {
            "p_data": self.p_data,
            "p_monitor": self.p_mon,

            "QBER_data": qber_data,
            "RawRate_in_data": raw_in_data,         # per signal pulse (conditional on data branch)
            "RawRate_total_sent": raw_total_sent,   # overall per pulse sent = p_data × P(signal) × raw_in_data
            
            "S_monitor": mon_out["S_value"],
            "SKR": skr,  # اضافه شد: نرخ کلید امن

            "E_A0_B0": mon_out["E_A0_B0"],
            "E_A0_B1": mon_out["E_A0_B1"],
            "E_A1_B0": mon_out["E_A1_B0"],
            "E_A1_B1": mon_out["E_A1_B1"],

            "monitor_timing": {
                "sigma_total_ps": mon_out["sigma_total_ps"],
                "F_mid": mon_out["F_mid"],
                "F_E2M": mon_out["F_E2M"],
                "F_L2M": mon_out["F_L2M"],
            }
        }

        if verbose:
            print("\n--- Hybrid Summary ---")
            print(f"Branch probabilities: Data={out['p_data']:.2f}, Monitor={out['p_monitor']:.2f}")
            print(f"DATA:    QBER = {out['QBER_data']*100:.4f} %")
            print(f"DATA:    RawRate (per signal pulse in data) = {out['RawRate_in_data']:.4e}")
            print(f"SIGNAL:  P(signal) = P('0')+P('1') = {p_signal:.4f}")
            print(f"OVERALL: RawRate (per pulse sent)           = {out['RawRate_total_sent']:.4e}  [= p_data × P(signal) × raw_in_data]")
            print(f"MON:     S = {out['S_monitor']:.6f}  (CHSH)")
            print(f"SKR:     Secure Key Rate = {out['SKR']:.4e}")
            if out["S_monitor"] > 2:
                viol = ((out["S_monitor"] - 2.0) / 2.0) * 100.0
                print(f"         ✓ CHSH violation: {viol:.2f}%")
            else:
                print("         ✗ No CHSH violation")
            if out["SKR"] > 0:
                print("         ✓ Secure key can be extracted")
            else:
                print("         ✗ No secure key (SKR <= 0)")

        return out

    def sweep_distance(self, distances_km: np.ndarray) -> Dict[str, np.ndarray]:
        qber = []
        raw_in_data = []
        raw_total = []
        S = []
        skr = []  # لیست برای ذخیره SKR

        # ثابت از روی آلیس (برای کل sweep)
        p_signal = self.alice.probs['0'] + self.alice.probs['1']

        for L in distances_km:
            p = self.params.copy()
            p['fiber_length_km'] = float(L)

            data_sim = DataLineQKD(p, self.alice)
            mon_sim = MonitoringCHSH(p, self.alice)

            q, r = data_sim.run(verbose=False)
            mon_out = mon_sim.analytical_CHSH()
            
            # محاسبه SKR برای این فاصله
            raw_total_sent_val = self.p_data * p_signal * r
            skr_val = compute_skr(mon_out["S_value"], q, raw_total_sent_val)

            qber.append(q * 100.0)  # به درصد تبدیل می‌کنیم
            raw_in_data.append(r)
            raw_total.append(raw_total_sent_val)
            S.append(mon_out["S_value"])
            skr.append(skr_val)

        return {
            "distances_km": distances_km,
            "qber_percent": np.array(qber),
            "raw_in_data": np.array(raw_in_data),
            "raw_total_sent": np.array(raw_total),
            "S_value": np.array(S),
            "SKR": np.array(skr)  # اضافه شد
        }

    def plot_all_vs_distance(self, L_max_km: float = 300.0, n_points: int = 101):
        """
        نمودارها به‌صورت جداگانه رسم می‌شوند (چهار شکل مستقل).
        """
        distances = np.linspace(0.0, float(L_max_km), int(n_points))
        res = self.sweep_distance(distances)

        # 1) QBER
        plt.figure(figsize=(7, 5))
        plt.plot(res["distances_km"], res["qber_percent"], linewidth=2.2, color='red')
        #plt.axhline(11.0, linestyle=":", alpha=0.6, label="~11% safety line", color='red')
        plt.ylabel("QBER (%)", fontweight="bold")
        plt.xlabel("Distance (km)", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        plt.margins(x=0)
        #plt.legend()
        #plt.title("Quantum Bit Error Rate vs. Distance")
        plt.tight_layout()

        # 2) Raw rates
        plt.figure(figsize=(7, 5))
        plt.semilogy(res["distances_km"], res["raw_in_data"], linewidth=2.2, 
                     label="Raw rate (per signal pulse in data)", color='green')
        plt.semilogy(res["distances_km"], res["raw_total_sent"], linewidth=2.2, linestyle="--",
                     label=f"Raw rate (per pulse sent) = p_data × P(signal) × data", color='orange')
        plt.ylabel("Raw Rate (log scale)", fontweight="bold")
        plt.xlabel("Fiber Length (km)", fontweight="bold")
        plt.grid(True, which="both", alpha=0.3, linestyle=":")
        plt.legend()
        plt.title("Raw Key Generation Rate")
        plt.xlim(left=0)
        plt.margins(x=0)
        plt.tight_layout()

        # 3) S (CHSH)
        plt.figure(figsize=(7, 5))
        plt.plot(res["distances_km"], res["S_value"], linewidth=2.2, 
                 label="S parameter", color='purple')
        plt.axhline(2.0, linestyle="--", linewidth=1.8, 
                    label="Classical limit = 2", color='red', alpha=0.7)
        plt.axhline(2.0 * math.sqrt(2.0), linestyle=":", linewidth=1.8, 
                    label=f"Quantum limit = {2*math.sqrt(2):.3f}", color='darkgreen', alpha=0.7)
        plt.xlabel("Distance (km)", fontweight="bold")
        plt.ylabel("S (CHSH)", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        #plt.title("CHSH Parameter (Bell Inequality)")
        plt.xlim(left=0)
        plt.margins(x=0)
        plt.tight_layout()

        # 4) SKR
        plt.figure(figsize=(7, 5))
        plt.semilogy(res["distances_km"], res["SKR"], linewidth=2.5, color='black')
        plt.xlabel("Distance (km)", fontweight="bold")
        plt.ylabel("SKR (per pulse)", fontweight="bold")
        plt.grid(True, which="both", alpha=0.3, linestyle=":")
        plt.ylim(bottom=1e-9)
        plt.xlim(left=0)
        plt.margins(x=0)
        # پیدا کردن جایی که SKR مثبت است
        #positive_skr_indices = np.where(res["SKR"] > 0)[0]
        #if len(positive_skr_indices) > 0:
            #max_distance = res["distances_km"][positive_skr_indices[-1]]
            #plt.axvline(max_distance, linestyle='--', alpha=0.5, color='gray',
                        #label=f'Max distance with SKR>0: {max_distance:.1f} km')
        #plt.legend(loc="upper right")
        #plt.title("Secure Key Rate")
        plt.tight_layout()
        
        plt.show()


# =========================================================
# بخش F) اجرا (نمونه‌ی آماده)
# =========================================================

if __name__ == "__main__":
    # یک دیکشنری پارامتر واحد که هر دو مدل بتوانند از آن استفاده کنند.
    # شما می‌توانید فقط پارامترهای ثانیه‌ای را بدهید؛ CHSH خودش به ps تبدیل می‌کند.
    user_params = {
        # مشترک
        "amplitude_a": 0.316,
        "loss_dB_per_km": 0.2,
        "fiber_length_km": 50.0,

        # احتمالات آلیس (اختیاری)
        "prob_0": 0.45,
        "prob_1": 0.45,
        "prob_d": 0.05,
        "prob_f": 0.05,

        # --------- Data-line params (واحد ثانیه)
        "detector_efficiency": 0.90,
        "dark_count_prob_per_gate": 1e-8,
        "after_pulse_prob": 0.00,

        "chromatic_dispersion": 17.0,      # ps/(nm·km)
        "spectral_width_nm": 0.0001,       # nm
        "pulse_fwhm_s": 1e-10,
        "jitter_fwhm_s": 3e-11,
        "bin_separation_s": 5e-10,
        "gate_width_s": 3e-10,

        # --------- Monitoring/CHSH params (اختیاری؛ اگر ندهید از بالا مشتق می‌شود)
        "eta_det_main": 0.90,
        "BS2_R": 0.85,
        # اگر اینها را بدهید، مستقیماً استفاده می‌شوند؛ اگر ندهید از fwhm_s تبدیل می‌کنیم:
        # "pulse_sigma_ps": 100.0,
        # "jitter_sigma_ps": 30.0,
        "dispersion_D_ps_nm_km": 17.0,
        "p_dark_main": 1e-8,
        "mu_bg_main": 0.0,
        "phi_mis": 0.2,
        "amp_imbalance": 0.02,
        "V_coh": 0.99,
        # اگر اینها را بدهید، مستقیم استفاده می‌شود؛ اگر ندهید از gate_width_s / bin_separation_s تبدیل می‌کنیم:
        # "gate_width_ps": 300.0,
        # "bin_separation_ps": 500.0,
    }

    sim = HybridBobRouter(user_params, p_data=0.90)

    # اجرای تک‌نقطه برای دیدن اعداد
    print("=" * 80)
    print("Single-point simulation at L = 50 km")
    print("=" * 80)
    results = sim.run_once(verbose=True)

    # رسم چهار نمودار مجزا بر حسب فاصله
    print("\n" + "=" * 80)
    print("Plotting each result vs distance in separate figures (including SKR)")
    print("=" * 80)
    sim.plot_all_vs_distance(L_max_km=300.0, n_points=151)
