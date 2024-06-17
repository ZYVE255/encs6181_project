import numpy as np

def body_to_nav(roll, pitch, yaw):
    """
    Returns the directed cosine matrix (DCM) to transform from the body frame to the navigation frame

    params:
    - roll: Vehicle roll in rads
    - pitch: Vehicle pitch in rads
    - yaw: Vehicle yaw in rads

    return:
    - dcm: DCM matrix
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    dcm = np.array([[cp*cy, -cr*sy+sr*sp*cy, sr*sy+cr*sp*cy],
                    [cp*sy, cr*cy+sr*sp*sy, -sr*cy+cr*sp*sy],
                    [-sp, sr*cp, cr*cp]])
    
    return dcm


def quat_to_euler(quat):
    """
    Converts a quaternion matrix [w, x, y, z] into an Euler angle [roll, pitch, yaw] matrix

    params:
    - quat: Quaternion matrix [w, x, y, z]

    return:
    - euler: Euler matrix [roll, pitch, yaw]
    """
    euler = np.zeros(shape=(quat.shape[0], 3))
    for i in range(quat.shape[0]):
        qw, qx, qy, qz = quat[i, :]
        roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
        pitch = -np.pi/2 + 2*np.arctan2(np.sqrt(1+2*(qw*qy-qx*qz)), np.sqrt(1-2*(qw*qy-qx*qz)))
        yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))
        euler[i,:] = [roll, pitch, yaw]
    return euler


def process_trajectory(traj, data_params, train_mode=False):
    """
    Processes trajectory and returns IMU, INS, and delta GPS data based on status of the data_params file

    params:
    - traj: Trajectory from a HDF5 file of the Mid-Air dataset
    - data_params: Dictionary of properties for the trajectory processing
    - train_mode: When true will save the mean and std of the requested items into the data_params dict for use in normalization

    returns:
    - gps_delta: Difference in gps position [dPx, dPy, dPz] of the trajectory
    - accel_data: Accelerometer data [ax, ay, az] of the trajectory
    - gyro_data: Gyroscope data [roll/s, pitch/s, yaw/s] of the trajectory
    - attitude_data: Attitude data [roll, pitch, yaw] of the trajectory
    - velocity_data: Velocity data [x, y, z] of the trajectory
    """

    """Data import from file"""
    # GPS
    gps_pos = traj['gps']['position']
    # Ground truth
    gnd_att = quat_to_euler(traj['groundtruth']['attitude'])
    gnd_vel = traj['groundtruth']['velocity']
    # IMU unit
    imu_acc = traj['imu']['accelerometer']
    imu_gyr = traj['imu']['gyroscope']

    """Sample high frequency IMU data to the same frequency as GPS"""
    # Get average IMU data for each timestep of GPS data (IMU 100Hz, GPS 1Hz)
    imu_sumdiv = int(np.floor(imu_acc.shape[0]/100))
    accel_data = np.zeros(shape=(imu_sumdiv+1,3))
    accel_data[0,:] = imu_acc[0,:]
    gyro_data = np.zeros(shape=(imu_sumdiv+1,3))
    gyro_data[0,:] = imu_gyr[0,:]

    for i in range(imu_sumdiv):
        accel_data[i+1,:] = np.average(imu_acc[1+100*i:1+100*(i+1),:], axis=0)
        gyro_data[i+1,:] = np.average(imu_gyr[1+100*i:1+100*(i+1),:], axis=0)
    
    """Sample attitude or integrate gyros"""
    if data_params["TRUE_ATT"]:
        # Get average attitude data for each timestep of GPS data (attitude 100Hz, GPS 1Hz)
        gnd_sumdiv = int(np.floor(gnd_att.shape[0]/100))
        attitude_data = np.zeros(shape=(gnd_sumdiv+1,3))
        attitude_data[0,:] = gnd_att[0,:]
        for i in range(gnd_sumdiv):
            attitude_data[i+1,:] = np.average(gnd_att[1+100*i:1+100*(i+1),:], axis=0)
    else:
        gyro_integration = np.zeros_like(gyro_data)
        gyro_integration[0,:] = gnd_att[0,:]
        for i in range(1, gyro_integration.shape[0]):
            gyro_integration[i,:] = gyro_integration[i-1,:] + (gyro_data[i,[1, 0, 2]] * [-1, 1, 1])
        # gyro_integration = gyro_integration[:, [1, 0, 2]] * [-1, 1, 1]
        attitude_data = gyro_integration

    """Transfor accelerometer from body to nav"""
    if data_params["ACCEL_TO_NAV"]:
        for i in range(accel_data.shape[0]):
            roll, pitch, yaw = attitude_data[i,:]
            accel_data[i,:] = body_to_nav(roll=roll, pitch=pitch, yaw=yaw) @ accel_data[i,:]
        accel_data = accel_data - [0, 0, -9.81]
    
    """Sample velocity or integrate accelerometers"""
    if data_params["TRUE_VEL"]:
        # Get average velocity data for each timestep of GPS data (attitude 100Hz, GPS 1Hz)
        gnd_sumdiv = int(np.floor(gnd_vel.shape[0]/100))
        velocity_data = np.zeros(shape=(gnd_sumdiv+1,3))
        velocity_data[0,:] = gnd_vel[0,:]
        for i in range(gnd_sumdiv):
            velocity_data[i+1,:] = np.average(gnd_vel[1+100*i:1+100*(i+1),:], axis=0)
    else:
        accel_integration = np.zeros_like(accel_data)
        accel_integration[0,:] = gnd_vel[0,:]
        for i in range(1, accel_integration.shape[0]):
            accel_integration[i,:] = accel_integration[i-1,:] + accel_data[i,:]
        velocity_data = accel_integration

    """Form GPS delta data"""
    # Change in GPS position
    gps_delta = gps_pos[1:,:] - gps_pos[:-1,:]
    gps_delta = np.vstack((np.zeros(shape=(1,3)), gps_delta))

    """Normalize input data"""
    if data_params["NORMALIZE_INPUT"]:
        if train_mode:
            # Accel
            acc_mean = np.mean(accel_data, axis=0)
            data_params["acc_mean"] = acc_mean
            acc_std  = np.std(accel_data, axis=0)
            data_params["acc_std"] = acc_std
            # Gyro
            gyr_mean = np.mean(gyro_data, axis=0)
            data_params["gyr_mean"] = gyr_mean
            gyr_std  = np.std(gyro_data, axis=0)
            data_params["gyr_std"] = gyr_std
            # Attitude
            att_mean = np.mean(attitude_data, axis=0)
            data_params["att_mean"] = att_mean
            att_std  = np.std(attitude_data, axis=0)
            data_params["att_std"] = att_std
            # Velocity
            vel_mean = np.mean(velocity_data, axis=0)
            data_params["vel_mean"] = vel_mean
            vel_std  = np.std(velocity_data, axis=0)
            data_params["vel_std"] = vel_std

            accel_data = (accel_data - acc_mean)/acc_std
            gyro_data = (gyro_data - gyr_mean)/gyr_std
            attitude_data = (attitude_data - att_mean)/att_std
            velocity_data = (velocity_data - vel_mean)/vel_std
        else:
            accel_data = (accel_data - data_params["acc_mean"])/data_params["acc_std"]
            gyro_data = (gyro_data - data_params["gyr_mean"])/data_params["gyr_std"]
            attitude_data = (attitude_data - data_params["att_mean"])/data_params["att_std"]
            velocity_data = (velocity_data - data_params["vel_mean"])/data_params["vel_std"]
    
    """Normalize output data"""
    if data_params["NORMALIZE_OUTPUT"]:
        if train_mode:
            gps_mean = np.mean(gps_delta, axis=0)
            data_params["gps_mean"] = gps_mean
            gps_std  = np.std(gps_delta, axis=0)
            data_params["gps_std"] = gps_std

            gps_delta = (gps_delta - gps_mean)/gps_std
        else:
            gps_delta = (gps_delta - data_params["gps_mean"])/data_params["gps_std"]

    return gps_delta, accel_data, gyro_data, attitude_data, velocity_data


def test_model(traj, model, data_params):
    """
    Runs the predictive model over the data of a given trajectory and returns the predictions

    params:
    - traj: Trajectory from a HDF5 file of the Mid-Air dataset
    - model: Model to be tested
    - data_params: Dictionary of properties for the trajectory processing

    return:
    - Y_pred_: Predicted GPS changes
    - gps_delta: True GPS changes
    """
    gps_delta, accel_data, gyro_data, attitude_data, velocity_data = process_trajectory(traj, data_params, train_mode=False)

    dp_ = gps_delta.shape[0]
    Y_pred_ = np.zeros(shape=(dp_,3))
    for i in range(dp_):
        if i <= data_params["MODEL_DELAY"]:
            # STILL NEED MORE DATA
            Y_pred_[i,:] = gps_delta[i,:]
        else:
            # ENOUGH DATA
            X_ = np.zeros(shape=(1,0))
            if data_params["INPUT_ACCL"]:
                X_ = np.hstack((X_, [accel_data[i-data_params["MODEL_DELAY"]-1:i-1,:].flatten()]))
            if data_params["INPUT_GYRO"]:
                X_ = np.hstack((X_, [gyro_data[i-data_params["MODEL_DELAY"]-1:i-1,:].flatten()]))
            if data_params["INPUT_ATTI"]:
                X_ = np.hstack((X_, [attitude_data[i-data_params["MODEL_DELAY"]-1:i-1,:].flatten()]))
            if data_params["INPUT_VELO"]:
                X_ = np.hstack((X_, [velocity_data[i-data_params["MODEL_DELAY"]-1:i-1,:].flatten()]))
            if data_params["INPUT_POUT"]:
                X_ = np.hstack((X_, [Y_pred_[i-data_params["MODEL_DELAY"]-1:i-1,:].flatten()]))
            Y_ = model(X_)
            Y_pred_[i,:] = Y_
    return Y_pred_, gps_delta