from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class KalmanFilter1D:
    def __init__(self, initial_value = 0, uncertainty=5**2, initial_variance=1):
        self.state_pred_val = lambda x: x
        self.state_pred_var = lambda p: p

        self.measurements = []

        self.uncertainty = uncertainty

        self.kalman_gain = lambda p, r: p/ (p + r)
        self.state_update_val = lambda x,z,k: x + k * (z - x)
        self.state_update_var = lambda p,k: p * (1 - k)
        self.kalman_gains = []

        self.value_estimates = [initial_value]
        self.variance_estimates = [initial_variance]

        self.next_val = None
        self.next_var = None

        self.n = 0


    def update(self, measurement: float) -> None:
        """
        Updates the estimated state using the measurement.
        Args:
            measurement (float): Measured position.
        """
        self.n += 1
        self.measurements.append(measurement)

        self.kalman_gains.append(self.kalman_gain(self.variance_estimates[-1], self.uncertainty))

        val_estimate = self.state_update_val(self.value_estimates[-1], measurement, self.kalman_gains[-1])
        var_estimate = self.state_update_var(self.variance_estimates[-1], self.kalman_gains[-1])

        self.next_val = self.state_pred_val(val_estimate)
        self.next_var = self.state_pred_var(var_estimate)

        self.value_estimates.append(val_estimate)
        self.variance_estimates.append(var_estimate)
    
    def plot(self, true_values: Optional[List[float]] = None, verbose: bool = True) -> None:
        """
        Plots measurements, estimates, and optional true value.

        Args:
            true_value (Optional[float]): The expected true value for comparison.
            verbose (bool): If True, shows the plot; otherwise, saves it to file.
        """
        self.n -= 1
        self.measurements = self.measurements[1:]
        true_values = true_values[1:]
        self.value_estimates = self.value_estimates[1:]
        self.variance_estimates = self.variance_estimates[1:]

        plt.figure(figsize=(15, 8))

        plt.title(label="Limited Kalman Filter for system with constant dynamic model")
        plt.plot(self.measurements, color='blue', marker='x', linestyle='dashed', linewidth=2, label="Measurements")
        plt.plot(true_values, color='green', marker='*', linestyle='--', linewidth=2, label="True values")
        plt.plot(self.value_estimates, color='red', marker='d', linestyle='dotted', linewidth=2, label="Estimates")

        x_est_lower = [norm.ppf(0.05, mu, np.sqrt(self.variance_estimates[i])) for i,mu in enumerate(self.value_estimates)]
        x_est_higher = [norm.ppf(0.95, mu, np.sqrt(self.variance_estimates[i])) for i,mu in enumerate(self.value_estimates)]
        plt.fill_between(range(nt), x_est_lower, x_est_higher, color='red', alpha=0.15, label="95% confidence interval")

        plt.xlabel("Iterations")
        plt.ylabel("Weight (g)")
        plt.legend()
        if verbose:
            plt.show()
        else:
            plt.savefig("KFConstantDynamic1DSystem.png")
            plt.close()

    def plot_kg(self):
        plt.figure(figsize=(15, 8))

        plt.title(label="Kalman Gain")
        plt.plot(self.kalman_gains, color='black', marker='x', linewidth=2, label="Measurements")

        plt.xlabel("Measurements")
        plt.ylabel("Kalman Gain")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    true_value = 50

    nt = 100
    r = 5**2

    values = [np.random.normal(true_value, np.sqrt(r)) for _ in range(nt)]
    true_values = [true_value] * len(values)

    kf = KalmanFilter1D(initial_value=60, uncertainty=r, initial_variance=225)

    for value in values:
        kf.update(value)
    
    kf.plot(true_values=true_values, verbose=True)
    kf.plot_kg()
    