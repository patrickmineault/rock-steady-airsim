# Command airsim
import airsim

import argparse
import datetime
import math
import matplotlib
import matplotlib.image
import numpy as np
import os
import pprint
import scipy
import scipy.misc
import sys
import tempfile
import time
from tqdm import tqdm

import tables
from tables import IsDescription, Float64Col

class Particle(IsDescription):
    x = Float64Col()
    y = Float64Col()      # Signed 64-bit integer
    z = Float64Col()      # Signed 64-bit integer
    speed = Float64Col()      # Signed 64-bit integer
    pitch = Float64Col()      # Signed 64-bit integer
    yaw = Float64Col()      # Signed 64-bit integer
    heading_pitch = Float64Col()      # Signed 64-bit integer
    heading_yaw = Float64Col()      # Signed 64-bit integer
    rotation_pitch = Float64Col()      # Signed 64-bit integer
    rotation_yaw = Float64Col()      # Signed 64-bit integer


def list_to_2d_uint8_array(flst, width, height):
    return np.reshape(np.asarray(flst, np.uint8), (height, width))

def draw_von_mises(A):
    # Via rejection sampling.
    while 1:
        theta = np.random.uniform(-np.pi, np.pi)
        a = np.exp(A * (np.cos(theta)-1))
        if a > np.random.rand():
            return theta

def main(args):
    client = airsim.VehicleClient()
    client.confirmConnection()

    ts = datetime.datetime.now().isoformat()[:-7].replace(':', '')
    tmp_dir = os.path.join(args.out_path, args.env, ts)

    print ("Saving images to %s" % tmp_dir)
    try:
        os.makedirs(tmp_dir)
    except OSError:
        if not os.path.isdir(tmp_dir):
            raise

    nseqs = 3600
    h5file = tables.open_file(os.path.join(tmp_dir, 'output.h5'), 
                        mode="w", 
                        title="Flythroughs")

    seq_len = 40
    short_seq_len = 10
    nominal_fps = 30

    labels_table = h5file.create_table('/', 'labels', Particle, expectedrows=nseqs)
    video_table = h5file.create_earray('/', 'videos', tables.UInt8Atom(), shape=(0, seq_len, 3, 112, 112), expectedrows=nseqs)
    short_video_table = h5file.create_earray('/', 'short_videos', tables.UInt8Atom(), shape=(0, short_seq_len, 3, 112, 112), expectedrows=nseqs)
    depth_table = h5file.create_earray('/', 'depth', tables.Float64Atom(), shape=(0, 112, 112), expectedrows=nseqs)
    

    for i in tqdm(range(nseqs)): # do few times

        # For flat environments, start ground plane localization not too high.
        ground_from = 5

        # 3 meters/s -> jogging speed
        MAX_SPEED = 3
        collisions = True
        pause = 0

        # Manually define areas of interest. Note that inside one of the airsim
        # environments, you can pull up coordinates using `;`. However, the 
        # coordinates listed are multiplied by 100 (i.e. the numbers are in cm
        # rather than meters); divide by 100 to define reasonable boundaries 
        # for the capture area.
        if args.env == 'blocks':
            x = np.random.uniform(-50, 50)
            y = np.random.uniform(-50, 50)
            pause = .05 # blocks is too fast sometimes
        elif args.env.startswith('nh'):
            x = np.random.uniform(-150, 150)
            y = np.random.uniform(-150, 150)
            client.simEnableWeather(True)
            for k in range(8):
                client.simSetWeatherParameter(k, 0.0)

            if args.env == 'nh_fall':
                client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 1.0)
                client.simSetWeatherParameter(airsim.WeatherParameter.RoadLeaf, 1.0)

            if args.env == 'nh_winter':
                client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 1.0)
                client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow, 1.0)
        elif args.env == 'mountains':
            # Most of the interesting stuff (e.g. the lake, roads) is on the 
            # diagonal of this very large environment (several kilometers!).
            xy = np.random.uniform(0, 2500)
            xmy = np.random.uniform(-100, 100)
            x = xy + xmy
            y = xy - xmy

            # This environment varies a lot in height, start from higher
            ground_from = 100  

        elif args.env == 'trap':
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)

            # This one has lots of branches, keep sequences that have collisions
            collisions = False  
        else:
            raise NotImplementedError(args.env)

        # Time of day effects works in blocks and trap only.
        time_of_day = np.random.randint(5, 21)
        
        client.simSetTimeOfDay(True, 
                               start_datetime = f"2020-03-21 {time_of_day:02}:00:00", 
                               is_start_datetime_dst = True, 
                               celestial_clock_speed = 0, 
                               update_interval_secs = 60, 
                               move_sun = True)

        if pause > 0:
            time.sleep(pause)

        pitch = np.random.uniform(-.25, .25) # Sometimes we look a little up, sometimes a little down
        roll = 0  # Should be 0 unless something is wrong

        # 360 degrees lookaround
        yaw = np.random.uniform(-np.pi, np.pi)

        heading_yaw = draw_von_mises(2.5)
        heading_pitch = draw_von_mises(16)

        # Max ~90 degrees per second head rotation
        rotation_yaw = 30 * np.pi / 180 * np.random.randn()
        rotation_pitch = 10 * np.pi / 180 * np.random.randn()
        speed = MAX_SPEED * np.random.rand()
        
        # Figure out the height of the ground. Move the camera way far above
        # ground, and get the distance to the ground via the depth buffer 
        # from the bottom camera.
        client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(x, y, -ground_from), 
                        airsim.to_quaternion(0, 0, 0)), True)

        if pause > 0:
            time.sleep(pause)

        responses = client.simGetImages([
            airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPlanner, pixels_as_float=True)
        ])
        response = responses[0]
        the_arr = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)

        # Then move to ~5.5 feet above the ground
        rgy = range(int(.4 * the_arr.shape[0]),
                    int(.6 * the_arr.shape[0]))
        rgx = range(int(.4 * the_arr.shape[0]),
                    int(.6 * the_arr.shape[0]))                    
        the_min = np.median(the_arr[rgy, rgx])

        z = the_min - ground_from - np.random.uniform(1.4, 2)
        
        if z > 50:
            # More than 50 meters below sea level, bad draw.
            continue

        #client.startRecording()
        z = z.item()

        depths = []
        frames = []
        
        booped = False
        for k in range(seq_len):
            if booped:
                continue

            # In nominal seconds
            t = (k - (seq_len - 1) / 2) / nominal_fps
            d = t * speed

            client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(x + d * np.cos(yaw + heading_yaw) * np.cos(pitch + heading_pitch), 
                                    y + d * np.sin(yaw + heading_yaw) * np.cos(pitch + heading_pitch), 
                                    z + d * np.sin(pitch + heading_pitch)), 
                    airsim.to_quaternion(pitch + t * rotation_pitch, 
                                         roll, 
                                         yaw + t * rotation_yaw)
                ), True)

            responses = client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanner, True, False),
            ])

            for j, response in enumerate(responses):
                if j == 0:
                    frames.append(
                        airsim.string_to_uint8_array(response.image_data_uint8).reshape(response.height, response.width, 3)[:, :, ::-1]
                    )

                if j == 1:
                    zbuff = airsim.list_to_2d_float_array(response.image_data_float, 
                                                          response.width, 
                                                          response.height)
                    depths.append(zbuff)

                    # Did we bump into something?
                    if collisions:
                        closest = zbuff[zbuff.shape[0]//4:-zbuff.shape[0]//4, 
                                        zbuff.shape[1]//4:-zbuff.shape[1]//4].min()
                        if closest < .5:
                            # oops I booped it again.
                            booped = True

                if j == 0 and args.save_images:
                    filename = os.path.join(tmp_dir, f"{i:02}_{k:02}.png")
                    matplotlib.image.imsave(filename, frames[-1])

            if pause > 0:
                time.sleep(pause)
        
        row = labels_table.row
        if not booped and not args.save_images:
            # Let's save this!
            row['x'] = x
            row['y'] = y
            row['z'] = z
            row['yaw'] = yaw
            row['pitch'] = pitch
            row['speed'] = speed
            row['heading_yaw'] = heading_yaw
            row['heading_pitch'] = heading_pitch
            row['rotation_yaw'] = rotation_yaw
            row['rotation_pitch'] = rotation_pitch
            row.append()
            
            V = np.stack(frames, axis=0).transpose((0, 3, 1, 2))
            V = V.reshape((1, ) + V.shape)
            video_table.append(V)

            # Take a subset of the data.
            mid_seq = range((seq_len - short_seq_len) // 2, 
                            (seq_len + short_seq_len) // 2)

            assert V.shape[1] == seq_len

            short_video_table.append(V[:, mid_seq, :, :, :])

            # Only record the mid-point z buffer
            n = seq_len // 2
            D = depths[n]
            D = D.reshape((1, ) + D.shape)
            depth_table.append(D)

    h5file.close()

    # currently reset() doesn't work in CV mode. Below is the workaround
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate flythroughs of different environments.')
    parser.add_argument('--env', type=str, default='', help='Environment to use')
    parser.add_argument('--save_images', default=False, action='store_true', help='Save images')
    parser.add_argument('--out_path', default='../../data/raw/', help='Where to save images and hdf5')
    args = parser.parse_args()
    main(args)