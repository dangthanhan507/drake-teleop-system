import os

for i in range(0, 1):
    source_intrinsics = 'camera_intrinsics.json'
    source_extrinsics = 'camera_extrinsics.json'
    destination_intrinsics = f'/data/andang/dataset/dataset_bottle_pickup_raised/demo_{ "{:03d}".format(i) }/camera_intrinsics.json'
    destination_extrinsics = f'/data/andang/dataset/dataset_bottle_pickup_raised/demo_{ "{:03d}".format(i) }/camera_extrinsics.json'
    os.system(f'cp {source_intrinsics} {destination_intrinsics}')
    os.system(f'cp {source_extrinsics} {destination_extrinsics}')
    # os.system(f'rm {destination}')