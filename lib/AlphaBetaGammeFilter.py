from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class StateEstimatorStatic1DSystem:
    """
    Implements a 1D estimation algorithm with state update, measurement tracking,
    prediction, file I/O, and visualization support.
    """

    def __init__(self, initial_state: float):
        """
        Initialize the estimator with an initial scalar state.

        Args:
            initial_state (float): The starting value of the estimated state.
        """
        self.state: float = initial_state
        self.values: List[float] = []

    def add_measurement(self, measurement: float) -> None:
        """
        Adds a new scalar measurement and updates the estimated state.

        Args:
            measurement (float): New scalar measurement.
        """
        self.values.append(measurement)
        self._update_state()

    def _update_state(self) -> None:
        """
        Updates the current estimated state as the mean of all measurements.
        """
        if self.values:
            self.state = float(np.mean(self.values))

    def predict(self, transition: float = 1.0, control_input: Optional[float] = None, control_gain: Optional[float] = None) -> None:
        """
        Predicts the next state using a basic linear model.

        Args:
            transition (float): Transition factor (like A in Kalman filters).
            control_input (Optional[float]): Control signal (u).
            control_gain (Optional[float]): Control gain (like B).
        """
        self.state *= transition
        if control_input is not None and control_gain is not None:
            self.state += control_gain * control_input

    def get_state(self) -> float:
        """
        Returns the current scalar estimated state.

        Returns:
            float: Estimated state.
        """
        return self.state

    def get_measurements(self) -> List[float]:
        """
        Returns the list of all past scalar measurements.

        Returns:
            List[float]: Measurement history.
        """
        return self.values

    def reset(self) -> None:
        """
        Clears all stored measurements and resets the state to 0.
        """
        self.state = 0.0
        self.values.clear()

    def save_to_file(self, filepath: Path) -> None:
        """
        Saves the state and measurements to a compressed .npz file.

        Args:
            filepath (Path): Path to save file to.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filepath,
            state=np.array([self.state]),
            values=np.array(self.values)
        )

    def load_from_file(self, filepath: Path) -> None:
        """
        Loads the state and measurements from a compressed .npz file.

        Args:
            filepath (Path): Path to load file from.
        """
        data = np.load(filepath)
        self.state = float(data["state"][0])
        self.values = data["values"].tolist()

    def plot_estimates(self, true_value: Optional[float] = None, verbose: bool = True) -> None:
        """
        Plots measurements, estimates, and optional true value.

        Args:
            true_value (Optional[float]): The expected true value for comparison.
            verbose (bool): If True, shows the plot; otherwise, saves it to file.
        """
        if not self.values:
            print("No measurements to plot.")
            return

        time = np.arange(len(self.values))
        measurements = np.array(self.values)
        estimates = np.array([np.mean(self.values[:i + 1]) for i in range(len(self.values))])

        plt.figure(figsize=(10, 4))
        plt.plot(time, measurements, label="Measurements", linestyle="--", marker="o")
        plt.plot(time, estimates, label="Estimate", linestyle="-", marker="s")
        if true_value is not None:
            plt.hlines(true_value, xmin=0, xmax=len(self.values) - 1, label="True Value", color="green")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("1D Estimation Over Time")
        plt.grid(True)
        plt.legend()

        if verbose:
            plt.show()
        else:
            plt.savefig("StateEstimatorStatic1DSystem.png")
            plt.close()

class AlphaBetaStateEstimatorConstant1DSystem:
    """
    Alpha-Beta filter for tracking a 1D constant-velocity object (e.g., aircraft).
    """

    def __init__(self, initial_position: float, initial_velocity: float, alpha: float = 0.85, beta: float = 0.005, dt: float = 1.0):
        """
        Initialize the tracker with initial state and filter gains.

        Args:
            initial_position (float): Initial estimate of position.
            initial_velocity (float): Initial estimate of velocity.
            alpha (float): Gain for position correction.
            beta (float): Gain for velocity correction.
            dt (float): Time step between updates.
        """
        self.x = initial_position
        self.v = initial_velocity
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.is_predicting = False

        self.position_history: List[float] = [self.x]
        self.velocity_history: List[float] = [self.v]
        self.measurements: List[float] = []
        self.predicted_positions: List[float] = []
    
    def enable_prediction(self) -> None:
        """Turns on automatic prediction before update."""
        self.is_predicting = True

    def disable_prediction(self) -> None:
        """Turns off automatic prediction before update."""
        self.is_predicting = False

    def predict(self) -> None:
        """
        Predicts the next state based on constant velocity motion model.
        """
        predicted_position = self.x + self.v * self.dt
        self.predicted_positions.append(predicted_position)
        self.x = predicted_position


    def update(self, measurement: float) -> None:
        """
        Updates the estimated state using the measurement.
        If prediction is enabled, it is applied before update.

        Args:
            measurement (float): Measured position.
        """
        if self.is_predicting:
            self.predict()
        
        residual = measurement - self.x
        self.x += self.alpha * residual
        self.v += self.beta * residual

        self.position_history.append(self.x)
        self.velocity_history.append(self.v)
        self.measurements.append(measurement)

    def get_state(self) -> Tuple[float, float]:
        """
        Returns the current estimated position and velocity.

        Returns:
            Tuple[float, float]: (position, velocity)
        """
        return self.x, self.v
    
    def get_velocity_estimates(self) -> List[float]:
        """
        Returns the list of estimated velocities over time.

        Returns:
            List[float]: Estimated velocities.
        """
        return self.velocity_history

    def reset(self, position: float = 0.0, velocity: float = 0.0) -> None:
        """
        Resets the tracker state and history.

        Args:
            position (float): Reset position.
            velocity (float): Reset velocity.
        """
        self.x = position
        self.v = velocity
        self.position_history = [self.x]
        self.velocity_history = [self.v]
        self.measurements = []
        self.predicted_positions = []


    def plot(self, true_positions: Optional[List[float]] = None, verbose: bool = True) -> None:
        """
        Plots the estimated positions, measurements, predictions, and true values.

        Args:
            true_positions (Optional[List[float]]): List of ground truth positions.
            verbose (bool): If True, shows the plot; otherwise saves it.
        """
        time = range(len(self.position_history))

        plt.figure(figsize=(10, 4))
        plt.plot(time, self.position_history, label='Estimated Position', marker='s')
        if self.measurements:
            plt.plot(time[1:], self.measurements, label='Measurements', linestyle='--', marker='o')
        if self.predicted_positions:
            plt.plot(time[1:], self.predicted_positions, label='Predicted (Pre-Update)', linestyle=':', marker='x')
        if true_positions:
            plt.plot(time[:len(true_positions)], true_positions, label='True Position', color='green')

        plt.xlabel("Time Step")
        plt.ylabel("Position")
        plt.title("Alpha-Beta Filter Position Tracking")
        plt.grid(True)
        plt.legend()

        if verbose:
            plt.show()
        else:
            plt.savefig("StateEstimatorConstant1DSystem.png")
            plt.close()
    
    def plot_velocity(self, true_velocity: Optional[List[float]] = None, verbose: bool = True) -> None:
        """
        Plots the velocity estimates, predicted velocity, and optional true velocity.

        Args:
            true_velocity (Optional[List[float]]): Ground truth velocity over time.
            verbose (bool): If True, shows the plot; otherwise saves it to file.
        """
        time = range(len(self.velocity_history))
        plt.figure(figsize=(10, 4))

        # Estimated velocity (after update)
        plt.plot(time, self.velocity_history, label='Estimated Velocity', marker='s')

        # Predicted velocities based on predicted position - previous position
        if self.predicted_positions:
            predicted_velocities = [
                (self.predicted_positions[i] - self.position_history[i]) / self.dt
                for i in range(len(self.predicted_positions))
            ]
            plt.plot(time[1:], predicted_velocities, label='Predicted Velocity', linestyle=':', marker='x')

        # True velocity values
        if true_velocity is not None:
            clipped_true = true_velocity[:len(self.velocity_history)]
            plt.plot(time[:len(clipped_true)], clipped_true, label='True Velocity', color='green')

        plt.xlabel("Time Step")
        plt.ylabel("Velocity")
        plt.title("Velocity Estimate Over Time (Alpha-Beta Filter)")
        plt.grid(True)
        plt.legend()

        if verbose:
            plt.show()
        else:
            plt.savefig("AlphaBetaStateEstimatorConstant1DSystemVsAccelerating.png")
            plt.close()

class AlphaBetaGammaStateEstimatorAccelerated1DSystem:
    """
    Alpha-Beta-Gamma filter for 1D systems with acceleration.
    Implements prediction and update using user-defined motion and update equations.
    """

    def __init__(self, initial_position: float, initial_velocity: float, initial_acceleration: float,
                 alpha: float = 0.85, beta: float = 0.005, gamma: float = 0.0001, dt: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        self.is_predicting = False

        # State variables
        self.x = initial_position
        self.v = initial_velocity
        self.a = initial_acceleration

        # Histories
        self.position_history = [self.x]
        self.velocity_history = [self.v]
        self.acceleration_history = [self.a]
        self.predicted_positions = [self.x]
        self.predicted_velocities: List[float] = []
        self.measurements = []

        # Lambda functions for motion model
        self.state_est_distance = lambda x, t, v: x + t * v
        self.state_est_velocity = lambda v, t, a: v + t * a
        self.state_est_acceleration = lambda a: a

        self.state_update_distance = lambda x, z: x + self.alpha * (z - x)
        self.state_update_velocity = lambda v, x, z, t: v + self.beta * ((z - x) / t)
        self.state_update_acceleration = lambda a, x, z, t: a + self.gamma * ((z - x) / (t ** 2))

    def enable_prediction(self) -> None:
        self.is_predicting = True

    def disable_prediction(self) -> None:
        self.is_predicting = False

    def predict(self) -> None:
        """
        Predicts the next state based on constant acceleration model.
        This prediction is stored for plotting.
        """
        predicted_x = self.state_est_distance(self.x, self.dt, self.v)
        self.predicted_positions.append(predicted_x)
        self.x = predicted_x
        self.v = self.state_est_velocity(self.v, self.dt, self.a)
        self.predicted_velocities.append(self.v)
        self.a = self.state_est_acceleration(self.a)

    def update(self, measurement: float) -> None:
        """
        Updates state using the measurement and prediction logic defined above.
        """
        if self.is_predicting:
            self.predict()

        residual = measurement - self.x
        self.x = self.state_update_distance(self.x, measurement)
        self.v = self.state_update_velocity(self.v, self.x, measurement, self.dt)
        self.a = self.state_update_acceleration(self.a, self.x, measurement, self.dt)

        self.position_history.append(self.x)
        self.velocity_history.append(self.v)
        self.acceleration_history.append(self.a)
        self.measurements.append(measurement)

    def get_state(self) -> Tuple[float, float, float]:
        return self.x, self.v, self.a

    def reset(self, position: float = 0.0, velocity: float = 0.0, acceleration: float = 0.0) -> None:
        self.x = position
        self.v = velocity
        self.a = acceleration
        self.is_predicting = False
        self.position_history = [self.x]
        self.velocity_history = [self.v]
        self.acceleration_history = [self.a]
        self.predicted_positions = [self.x]
        self.measurements = []

    def plot(self, true_positions: Optional[List[float]] = None, verbose: bool = True) -> None:
        time = range(len(self.position_history))
        plt.figure(figsize=(10, 4))
        plt.plot(time, self.position_history, label='Estimated Position', marker='s')
        if self.measurements:
            plt.plot(time[1:], self.measurements, label='Measurements', linestyle='--', marker='o')
        if self.predicted_positions[1:]:
            plt.plot(time[1:], self.predicted_positions[1:], label='Predicted Position', linestyle=':', marker='x')
        if true_positions:
            plt.plot(time[:len(true_positions)], true_positions, label='True Position', color='green')
        plt.xlabel("Time Step")
        plt.ylabel("Position")
        plt.title("Alpha-Beta-Gamma Filter Position Tracking")
        plt.grid(True)
        plt.legend()
        if verbose:
            plt.show()
        else:
            plt.savefig("alpha_beta_gamma_position_plot.png")
            plt.close()

    def plot_velocity(self, true_velocity: Optional[List[float]] = None, verbose: bool = True) -> None:
        time = range(len(self.velocity_history))
        plt.figure(figsize=(10, 4))
        plt.plot(time, self.velocity_history, label='Estimated Velocity', marker='s')
        if self.predicted_velocities:
            plt.plot(time[1:], self.predicted_velocities, label='Predicted Velocity', linestyle=':', marker='x')
        if true_velocity:
            plt.plot(time[:len(true_velocity)], true_velocity, label='True Velocity', color='green')
        plt.xlabel("Time Step")
        plt.ylabel("Velocity")
        plt.title("Alpha-Beta-Gamma Filter Velocity Tracking")
        plt.grid(True)
        plt.legend()
        if verbose:
            plt.show()
        else:
            plt.savefig("alpha_beta_gamma_velocity_plot.png")
            plt.close()

    def plot_acceleration(self, true_acceleration: Optional[List[float]] = None, verbose: bool = True) -> None:
        time = range(len(self.acceleration_history))
        plt.figure(figsize=(10, 4))
        plt.plot(time, self.acceleration_history, label='Estimated Acceleration', marker='s')
        if len(self.acceleration_history) > 1:
            predicted_acc = self.acceleration_history[:-1]
            plt.plot(time[1:], predicted_acc, label='Predicted Acceleration', linestyle=':', marker='x')
        if true_acceleration:
            plt.plot(time[:len(true_acceleration)], true_acceleration, label='True Acceleration', color='green')
        plt.xlabel("Time Step")
        plt.ylabel("Acceleration")
        plt.title("Alpha-Beta-Gamma Filter Acceleration Tracking")
        plt.grid(True)
        plt.legend()
        if verbose:
            plt.show()
        else:
            plt.savefig("alpha_beta_gamma_acceleration_plot.png")
            plt.close()

        
if __name__ == "__main__":
    choice = "AlphaBetaGammaStateEstimatorAccelerated1DSystem"
    if choice == "StateEstimatorStatic1DSystem":
        est = StateEstimatorStatic1DSystem(initial_state=np.zeros(3))

        # Add measurements
        est.add_measurement(996)
        est.add_measurement(994)
        est.add_measurement(1021)
        est.add_measurement(1000)
        est.add_measurement(1002)
        est.add_measurement(1010)
        est.add_measurement(983)
        est.add_measurement(971)
        est.add_measurement(993)
        est.add_measurement(1023)

        est.plot_estimates(true_value=np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]), verbose=True)
    elif choice == "AlphaBetaStateEstimatorConstant1DSystem":
        est = AlphaBetaStateEstimatorConstant1DSystem(initial_position=30000, initial_velocity=40, dt=5)

        # Add measurements
        est.enable_prediction()
        est.update(30171)
        est.update(30353)
        est.update(30756)
        est.update(30799)
        est.update(31018)
        est.update(31278)
        est.update(31276)
        est.update(31379)
        est.update(31748)
        est.update(32175)

        est.plot(true_positions=[30000, 30200, 30400, 30600, 30800, 31000, 31200, 31400, 31600, 31800, 32000], verbose=True)
    
    elif choice == "AlphaBetaStateEstimatorConstant1DSystemVsAccelerating":
        est = AlphaBetaStateEstimatorConstant1DSystem(initial_position=30000, initial_velocity=50, dt=5)

        # Add measurements
        est.enable_prediction()
        est.update(30221)
        est.update(30453)
        est.update(30906)
        est.update(30999)
        est.update(31368)
        est.update(31978)
        est.update(32526)
        est.update(33379)
        est.update(34698)
        est.update(36275)

        est.plot(true_positions=[30000, 30250, 30500, 30750, 31000, 31450, 32050, 33050, 34450, 36250, 38450], verbose=True)
        est.plot_velocity(true_velocity=[50, 50, 50, 50, 90, 130, 170, 210, 250, 290], verbose=True)
    
    elif choice == "AlphaBetaGammaStateEstimatorAccelerated1DSystem":
        est = AlphaBetaGammaStateEstimatorAccelerated1DSystem(initial_position=30000, initial_velocity=50, initial_acceleration=0, dt=5, alpha=0.2, beta= 0.1, gamma=0.1)

        # Add measurements
        est.enable_prediction()
        est.update(30221)
        est.update(30453)
        est.update(30906)
        est.update(30999)
        est.update(31368)
        est.update(31978)
        est.update(32526)
        est.update(33379)
        est.update(34698)
        est.update(36275)

        est.plot(true_positions=[30000, 30250, 30500, 30750, 31000, 31450, 32050, 33050, 34450, 36250, 38450], verbose=True)
        est.plot_velocity(true_velocity=[50, 50, 50, 50, 90, 130, 170, 210, 250, 290], verbose=True)
        est.plot_acceleration(true_acceleration=[0,0,0,0,8,8,8,8,8,8])
