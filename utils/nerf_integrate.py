import subprocess
import threading
import time
import socket
import struct
import pickle
import torch
import sys

def start_server(nerfstudio_python, server_script):
    server_process = subprocess.Popen(
        [nerfstudio_python, server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return server_process

def wait_for_server_ready(server_process):
    while True:
        line = server_process.stdout.readline()
        if not line:
            raise RuntimeError("Server process terminated unexpectedly.")
        print(line.strip())
        if "localhost" in line:
            print("Server is ready.")
            break

def send_pose_and_receive_depth(ros_poses, host='localhost', port=50007):
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
    depth = pickle.loads(output_data)

    s.close()
    return depth

def main():
    nerfstudio_python = '/home/qianzhong/anaconda3/envs/nerfstudio/bin/python'
    server_script = '/home/qianzhong/com_test_nerf/nerf_server.py'

    server_process = start_server(nerfstudio_python, server_script)
    wait_for_server_ready(server_process)

    try:
        stdout, stderr = server_process.communicate(timeout=1)
        if stdout:
            print(f"Server stdout:\n{stdout}")
        if stderr:
            print(f"Server stderr:\n{stderr}")
        pose = torch.eye(4)[None, :3, ...]
        depth,img = send_pose_and_receive_depth(pose)
        print("Received depth:", depth)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        server_process.terminate()
        server_process.wait()
        print("Server process terminated.")

if __name__ == "__main__":
    main()
