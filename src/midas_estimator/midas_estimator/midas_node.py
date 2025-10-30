#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class MidasDepthEstimator(Node):
    def __init__(self):
        super().__init__('midas_depth_estimator_node')

        # --- 1. Load MiDaS Model ---
        self.get_logger().info('Loading MiDaS model...')
        # Use 'MiDaS_small' for better speed on CPU
        # Use 'MiDaS' for better accuracy (slower)
        model_type = "MiDaS_small" 
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # Set device (use GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)
        self.midas.eval() # Set model to evaluation mode

        # Load the corresponding transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform if "small" in model_type else midas_transforms.dpt_transform

        self.get_logger().info(f'MiDaS model "{model_type}" loaded on {self.device}.')

        # --- 2. Define QoS Profile ---
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, # Match your webcam publisher
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- 3. Create Subscriber and Publisher ---
        self.image_topic = '/image_raw' # Topic from your Pi
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile)

        # We'll publish the colorized depth map for easy viewing
        self.depth_publisher_ = self.create_publisher(Image, '/midas/depth_image', qos_profile)
        self.cv_bridge = CvBridge()

        # --- 4. Create OpenCV Window ---
        self.window_name = "MiDaS Depth Estimation"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(f'Subscribing to {self.image_topic}...')

    def image_callback(self, msg):
        self.get_logger().debug('Image received...')
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        if cv_image is None:
            return

        # --- 5. Process Image with MiDaS ---
        try:
            # MiDaS expects RGB, OpenCV gives BGR
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Apply model-specific transforms
            input_batch = self.transform(img_rgb).to(self.device)

            with torch.no_grad(): # Disable gradient calculation for inference
                prediction = self.midas(input_batch)

                # Resize prediction to match input image size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Get the depth map as a numpy array
            depth_map = prediction.cpu().numpy()

            # --- 6. Visualize and Publish Depth Map ---

            # Normalize the depth map for visualization (0-255)
            # MiDaS gives inverse depth, so normalize it
            depth_map_viz = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
            depth_map_viz = (depth_map_viz * 255).astype(np.uint8)

            # Apply a colormap (like VIRIDIS) to make it easy to see
            depth_colormap = cv2.applyColorMap(depth_map_viz, cv2.COLORMAP_VIRIDIS)

            # Display the colorized depth map
            cv2.imshow(self.window_name, depth_colormap)
            cv2.waitKey(1)

            # Publish the colorized depth map as a ROS Image
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_colormap, "bgr8")
            depth_msg.header.stamp = msg.header.stamp # Use the same timestamp as input
            depth_msg.header.frame_id = "midas_depth_frame"
            self.depth_publisher_.publish(depth_msg)

        except Exception as e:
            self.get_logger().error(f'Error during MiDaS processing: {e}')

    def destroy_node(self):
        cv2.destroyAllWindows()
        self.get_logger().info('MiDaS node shutting down.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    midas_estimator = None
    try:
        midas_estimator = MidasDepthEstimator()
        rclpy.spin(midas_estimator)
    except Exception as e: # Catch startup errors (like no internet)
        if midas_estimator:
            midas_estimator.get_logger().fatal(f'Node crashed: {e}')
        else:
            print(f'Node failed to initialize: {e}')
    except KeyboardInterrupt:
        pass
    finally:
        if midas_estimator:
            midas_estimator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()