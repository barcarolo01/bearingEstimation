import numpy as np


def load_imu_model(floater,model_name,dt=1.0):
    if model_name == 'ADIS16470':
        # Accelerometer
        floater.sigma_accel_bias = np.ones(3) * (4e-3)
        floater.accel_bias = np.random.normal(0,floater.sigma_accel_bias)
        floater.sigma_accel_white_noise = np.ones(3) * (0.037 / np.sqrt(3600 * dt))
        floater.sigma_accel_bias_driving = np.ones(3) * (13e-6 * 9.81 * np.sqrt(dt / 200.0))

        # Gyroscope
        floater.sigma_gyro_bias = np.deg2rad(0.2)
        floater.gyro_bias = np.random.normal(0, floater.sigma_gyro_bias)
        floater.sigma_gyro_white_noise = np.deg2rad(0.34) / np.sqrt(3600 * dt)
        floater.sigma_gyro_bias_driving = np.deg2rad(8.0 / 3600.0) * np.sqrt(dt / 200) 

        

    elif model_name == 'HG4930': # It is assumed an observation time of 200 seconds
        # Accelerometer
        floater.sigma_accel_bias = np.ones(3) * (1.7e-3)
        floater.sigma_accel_white_noise = np.ones(3) * (0.03 / np.sqrt(3600 * dt))
        floater.sigma_accel_bias_driving = np.ones(3) * (25e-6 * np.sqrt(dt / 200.0))

        # Gyroscope
        floater.sigma_gyro_bias = np.deg2rad(7 / 3600)
        floater.sigma_gyro_white_noise = np.deg2rad(0.04) / np.sqrt(3600 * dt)
        floater.sigma_gyro_bias_driving = np.deg2rad(0.25 / 3600.0) * np.sqrt(dt / 200) 

    else:
        raise ValueError(f"Unknown model name '{model_name}'.")

    return floater
