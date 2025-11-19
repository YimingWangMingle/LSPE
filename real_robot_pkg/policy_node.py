#!/usr/bin/env python3
# coding: utf-8

# Basic imports
import os  
import threading  
import time  
import numpy as np

import rclpy
from rclpy.node import Node

# ROS2 msgs imports 
from sensor_msgs.msg import CompressedImage
from unitree_go.msg import WirelessController 

# CV2 imports
import cv2  
from cv_bridge import CvBridge 

# PyTorch imports
import torch  
import torch.nn as nn  
import torchvision.transforms as transforms
import torchvision.models as models  
from PIL import Image as PILImage  

class PolicyCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class PolicyControlNode(Node):  
    def __init__(self):
        super().__init__('policy_control_node')

        self.declare_parameter('speed_scale', 0.5)
        self.speed_scale = self.get_parameter('speed_scale').value
        
        self.declare_parameter('camera_fps', 10.0)
        self.camera_fps = self.get_parameter('camera_fps').value
        
        self.declare_parameter('infer_hz', 10.0)
        self.infer_hz = self.get_parameter('infer_hz').value
        self.action_pub = self.create_publisher(WirelessController, '/wirelesscontroller', 10)
        self.cam_pub = self.create_publisher(CompressedImage, '/cam_image', 5)
        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/cam_image', 
            self.image_callback, 
            10
        )

        gstreamer_str = "udpsrc address=230.1.1.1 port=1720 multicast-iface=eth0 ! application/x-rtp, media=video, encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,width=1280,height=720,format=BGR ! appsink drop=1"
        self.cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open GStreamer camera pipeline!")

        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

        self.policy_timer = self.create_timer(1.0/self.infer_hz, self.policy_callback)
        
        self.bridge = CvBridge() 
        self._lock = threading.Lock()  
        self.latest_image_pil = None 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PolicyCNN(num_classes=7).to(self.device)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'policy_model.pth')
        if not os.path.isfile(model_path):
            self.get_logger().error(f"Cannot find model file: {model_path}")
        else:
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.get_logger().info(f"Loaded policy model from {model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load model: {e}")
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        self.get_logger().info(
            f"PolicyControlNode started: camera@{self.camera_fps:.1f}Hz, infer@{self.infer_hz:.1f}Hz"
        )
    
    def _camera_loop(self):
        interval = 1.0 / self.camera_fps
        
        while rclpy.ok():
            start_time = time.time()
            
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    msg = CompressedImage()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.format = "jpeg"
                    msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
                    self.cam_pub.publish(msg)
                else:
                    self.get_logger().warn("Camera read failed", throttle_duration_sec=5.0)

            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
        
        self.cap.release()

    def image_callback(self, msg):
        try:

            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb_img)
            with self._lock:
                self.latest_image_pil = pil_img
        except Exception as e:
            self.get_logger().error(f"image_callback conversion error: {str(e)}")

    def policy_callback(self):
        with self._lock:
            img = self.latest_image_pil
        
        if img is None:
            self.get_logger().warn("No image received yet; skipping inference.", throttle_duration_sec=5.0)
            return
            
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)  
            action = torch.argmax(logits, dim=1).item()  
            
        self.get_logger().info(f"Inferred action: {action}", throttle_duration_sec=1.0)
        
        s = float(self.speed_scale)
        current_action = WirelessController()
        
        if action == 0:
            current_action.rx = +s   
        elif action == 1:
            current_action.rx = -s   
        elif action == 2:
            current_action.ly = +s  
        elif action == 3:
            current_action.ly = -s   
        elif action == 4:
            current_action.lx = +s   
        elif action == 5:
            current_action.lx = -s   
        elif action == 6:
            pass  
            
        self.action_pub.publish(current_action)

def main(args=None):
    try:
        rclpy.init(args=args)
        node = PolicyControlNode()
        rclpy.spin(node)
        node.destroy_node()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
