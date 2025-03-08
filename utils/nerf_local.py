from typing import Dict, List
from pathlib import Path
import os

import torch
import yaml
import pdb
import sys
# print(sys.path)

from nerfstudio.utils.eval_utils import eval_load_checkpoint
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.utils.poses import multiply
from torchvision.utils import save_image

# from rich.console import Console


class Nerf:
    def __init__(
        self,
        config_path: Path,
        data_path: Path,
        render_camera_ints: Dict[str, float],
        device='cuda:0',
        resolution_quality:float = 0.15
    ) -> None:
        
        self.device = device
        torch.device(self.device)
        
        self.ns_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

        self.ns_config.pipeline.datamanager._target = all_methods[
            self.ns_config.method_name
        ].pipeline.datamanager._target
        self.ns_config.load_dir = config_path.parent / "nerfstudio_models"
        self.ns_config.pipeline.datamanager.dataparser.data = data_path

        # self.console = console
        # self.console.rule("[bold green]Loading NeRF Checkpoint", style="bold blue")

        self.pipeline = self.ns_config.pipeline.setup(
            device=self.device, test_mode="inference"
        )
        self.pipeline.eval()
        _, _ = eval_load_checkpoint(self.ns_config, self.pipeline)
        self.dataparser_T = (
            self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        ).to(self.device)
        self.dataparser_scale = (
            self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        )

        assert "fx" in render_camera_ints, "fx not in render_camera_ints"
        assert "fy" in render_camera_ints, "fy not in render_camera_ints"
        assert "cx" in render_camera_ints, "cx not in render_camera_ints"
        assert "cy" in render_camera_ints, "cy not in render_camera_ints"
        assert "w" in render_camera_ints, "w not in render_camera_ints"
        assert "h" in render_camera_ints, "h not in render_camera_ints"
        self.render_camera = Cameras(
            camera_to_worlds=torch.eye(4)[None, :3, ...],
            fx=render_camera_ints["fx"],
            fy=render_camera_ints["fy"],
            cx=render_camera_ints["cx"],
            cy=render_camera_ints["cy"],
            width=render_camera_ints["w"],
            height=render_camera_ints["h"],
        )
        self.render_camera.height = self.render_camera.height.to(self.device)
        self.render_camera.width = self.render_camera.width.to(self.device)
        self.render_camera.fx = self.render_camera.fx.to(self.device)
        self.render_camera.fy = self.render_camera.fy.to(self.device)
        self.render_camera.cx = self.render_camera.cx.to(self.device)
        self.render_camera.cy = self.render_camera.cy.to(self.device)
        self.render_camera.camera_type = self.render_camera.camera_type.to(self.device)

        # self.render_camera.width = torch.tensor([[640]])  # Example of lower resolution width
        # self.render_camera.height = torch.tensor([[360]])  # Example of lower resolution height

        # if "rescale" in render_camera_ints:
        #     self.render_camera.rescale_output_resolution(render_camera_ints["rescale"])

        # TODO: change this part with better setting
        self.render_camera.rescale_output_resolution(torch.tensor([resolution_quality]).to(self.device))
        new_h = self.render_camera.height.item()
        new_w = self.render_camera.width.item()

        

    def render(self, pose: torch.Tensor) -> torch.Tensor:
        # Render from a single pose, get minimum depth with the 3D object
        # self.render_camera.rescale_output_resolution(0.5)
        self.render_camera.camera_to_worlds = pose
        # self.render_camera.width = torch.tensor([[640]])
        # self.render_camera.height = torch.tensor([[360]]
        
        # pdb.set_trace()
        outputs = self.pipeline.model.get_outputs_for_camera(self.render_camera)
        img = outputs["rgb"]
        depth = outputs["depth"] # requires_grad = false
        depth.requires_grad_(False)
        img.requires_grad_(False)
        # pdb.set_trace()
        # return torch.min(depth[0:int(float(self.render_camera.height[0][0])/2),:,:]), img
        return depth, img

    def convert_poses_quad2ns(self, ros_poses: torch.Tensor) -> torch.Tensor:
        """
        Convert from a quad pose convention to a Nerfstudio pose.
            1. Convert coordinate systems from ROS to Nerfstudio
            2. Transform with the Nerfstudio dataparser transform and scale

        Args:
            poses (torch.Tensor): expected shape (N, 3, 4)

        Returns:
            torch.Tensor: expected shape (N, 3, 4)
        """
        T_ns = ros_poses[:, :, [1, 2, 0, 3]]
        T_ns[:, :, [0, 2]] *= -1
        T_ns = multiply(self.dataparser_T, T_ns)
        T_ns[:, :, 3] *= self.dataparser_scale
        return T_ns

    def render_quad_poses(self, ros_poses: torch.Tensor) -> List[torch.Tensor]:
        """
        Render from a batch of quad poses

        Args:
            ros_poses (torch.Tensor): expected shape (N, 3, 4)
        Returns:

        """
        ns_poses = self.convert_poses_quad2ns(ros_poses)
        num_envs = ros_poses.size(0)
        depth_list = []
        img_list = []
        for i in range(ns_poses.shape[0]):
            pose = ns_poses[i].unsqueeze(0)
            # depth[i, :], img =self.render(pose)
            depth, img =self.render(pose)
            depth_list.append(depth)
            img_list.append(img)
        return torch.stack(depth_list,dim=0), torch.stack(img_list,dim=0)
    
    def to_device(self, device='cuda'):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))


if __name__ == "__main__":
    
    cwd = os.getcwd()
    nerf_path = Path(
        "/home/qianzhong/com_test_nerf/jv_simple_nerf/trained_nerfs/column2/2024-01-17_083853/config.yml"
    )
    
    data_path = Path("/home/qianzhong/com_test_nerf/jv_simple_nerf/nerf_training_data/columns2")
    pose = torch.eye(4)[None, :3, ...]

    render_camera_ints = {
        "fx": 636.76,
        "fy": 636.05,
        "cx": 646.82,
        "cy": 370.98,
        "w": 1280,
        "h": 720,
    }
    nerf = Nerf(nerf_path, data_path, render_camera_ints)
    images = nerf.render_quad_poses(pose)
    save_image(images[0], "../data/test.png")
