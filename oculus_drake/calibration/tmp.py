import os

dataset_path = '/data/andang/dataset/bottle_pickup'
num_demos = sum([1 for folder in os.listdir(dataset_path) if folder.startswith('demo_')])
for i in range(0, num_demos):
    source_intrinsics = 'camera_intrinsics.json'
    source_extrinsics = 'camera_extrinsics.json'
    destination_intrinsics = os.path.join(dataset_path, f'demo_{ "{:03d}".format(i) }', 'camera_intrinsics.json')
    destination_extrinsics = os.path.join(dataset_path, f'demo_{ "{:03d}".format(i) }', 'camera_extrinsics.json')
    os.system(f'cp {source_intrinsics} {destination_intrinsics}')
    os.system(f'cp {source_extrinsics} {destination_extrinsics}')
    # os.system(f'rm {destination}')