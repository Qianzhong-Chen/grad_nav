import socket
import struct
import pickle
import torch
import time

# need to turn on the nerf server at ~/com_test_nerf manually

def send_pose_and_receive_rgbd(ros_poses, host='localhost', port=50007):
    # Create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    # Serialize data
    data = pickle.dumps(ros_poses)

    # Send data length and data
    data_length = struct.pack('!I', len(data))
    s.sendall(data_length)
    s.sendall(data)

    # Receive output length
    output_length_bytes = s.recv(4)
    output_length = struct.unpack('!I', output_length_bytes)[0]

    # Receive output data
    output_data = b''
    while len(output_data) < output_length:
        packet = s.recv(output_length - len(output_data))
        if not packet:
            break
        output_data += packet

    # Deserialize output data
    rgbd = pickle.loads(output_data)

    s.close()
    return rgbd

if __name__ == "__main__":
    # Define robot pose (example)
    num_1 = 0
    num_2 = 0
    while num_1 < 10:
        # pose = torch.eye(4)[None, :3, ...]  # Shape: (1, 3, 4)
        pose = torch.rand(1,3,4)
        depth, img = send_pose_and_receive_rgbd(pose)
        print(f'num_1 {num_1}')
        print("Received depth:", depth)
        # print(img)
        num_1 += 1
    time.sleep(3)
    while num_2 < 10:
        # pose = torch.eye(4)[None, :3, ...]  # Shape: (1, 3, 4)
        pose = torch.rand(1,3,4)
        depth, img = send_pose_and_receive_rgbd(pose)
        print(f'num_2 {num_2}')
        print("Received depth:", depth)
        # print(img)
        num_2 += 1
