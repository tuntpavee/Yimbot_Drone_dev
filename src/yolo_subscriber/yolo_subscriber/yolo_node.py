#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import math # For distance calculation
# Import QoS settings
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# --- Variables for ROI and Tracking ---
drawing = False
roi_start_point = (-1, -1)
roi_end_point = (-1, -1)
roi_defined = False # Flag to indicate ROI selection is complete for one frame
target_track_id = None # Store the ID of the object to track
last_known_bbox = None # Store the last known bounding box of the tracked object
valid_roi_coords = None # Will store (x1, y1, x2, y2) of the selection box

# --- Mouse callback function ---
def handle_mouse_input(event, x, y, flags, param):
    global roi_start_point, roi_end_point, drawing, roi_defined, target_track_id, last_known_bbox, valid_roi_coords

    node = param
    frame_shape = node.current_frame_shape
    if frame_shape is None: return
    h, w = frame_shape[:2]

    # Clamp coordinates within callback to prevent errors during drawing
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_defined = False
        target_track_id = None # Clear previous target
        last_known_bbox = None
        valid_roi_coords = None
        roi_start_point = (x, y)
        roi_end_point = (x, y)
        node.get_logger().info(f'ROI Selection Start: {roi_start_point}')

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing: # Ensure this only triggers if we were actually drawing
            drawing = False
            roi_end_point = (x, y)
            # Check for minimal area
            x1 = min(roi_start_point[0], roi_end_point[0])
            y1 = min(roi_start_point[1], roi_end_point[1])
            x2 = max(roi_start_point[0], roi_end_point[0])
            y2 = max(roi_start_point[1], roi_end_point[1])
            # Clamp coordinates AFTER calculating box to ensure start point isn't clamped prematurely
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)

            if (x2 - x1) > 5 and (y2 - y1) > 5: # Require minimum size
                roi_defined = True # Signal to image_callback to find target
                valid_roi_coords = (x1, y1, x2, y2) # Store clamped ROI for find_target
                node.get_logger().info(f'ROI Selection End: Clamped [{x1},{y1}, {x2},{y2}]. Attempting target ID.')
            else:
                node.get_logger().warn('ROI too small, selection cancelled.')
                roi_start_point = (-1, -1)
                roi_end_point = (-1, -1)
                roi_defined = False
                target_track_id = None
                last_known_bbox = None
                valid_roi_coords = None


    elif event == cv2.EVENT_RBUTTONDOWN: # Right-click to clear target
         drawing = False
         roi_defined = False
         target_track_id = None
         last_known_bbox = None
         valid_roi_coords = None
         roi_start_point = (-1, -1)
         roi_end_point = (-1, -1)
         node.get_logger().info('Tracking Target Cleared.')

class YoloSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber_node')

        # --- Define QoS Profile ---
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Subscribe to the WEBCAM topic ---
        self.image_topic = '/image_raw' # For webcam setup
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile=qos_profile) # Use specified QoS
        self.subscription
        self.cv_bridge = CvBridge()
        # --- Load a TRACKING capable model ---
        # Use 'yolov8n.pt' for boxes, 'yolov8n-seg.pt' for masks
        self.model = YOLO('yolov8m.pt') 
        self.current_frame_shape = None

        # --- Create window and set mouse callback ---
        self.window_name = "YOLO Tracking (Drag ROI to Select, Right-click to clear)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, handle_mouse_input, param=self)

        self.get_logger().info(f'YOLO model loaded. Subscribing to {self.image_topic}...')
        self.get_logger().info('--- Drag Left Mouse Button to select object to track ---')
        self.get_logger().info('--- Right Mouse Button clears tracking target ---')

    # --- THIS IS THE UPDATED FUNCTION ---
    def find_target_id_in_roi(self, cv_image):
        global roi_start_point, roi_end_point, target_track_id, last_known_bbox, valid_roi_coords

        if valid_roi_coords is None:
            self.get_logger().warn('find_target_id_in_roi called with invalid coords.')
            return

        x1, y1, x2, y2 = valid_roi_coords
        self.get_logger().info(f'--- Finding Target ID in ROI [{x1},{y1}, {x2},{y2}] ---')

        roi_center_x = (x1 + x2) / 2
        roi_center_y = (y1 + y2) / 2

        self.get_logger().info('--- Running initial track on FULL frame ---')
        try:
            # --- MODIFICATION: Run track on the FULL image ---
            results = self.model.track(cv_image, persist=False, verbose=False) # Get initial IDs
        except Exception as e:
            self.get_logger().error(f"Error during initial track for ID selection: {e}")
            return

        min_dist = float('inf')
        selected_id = None
        selected_bbox = None

        # Check if tracking results exist and have IDs
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy() # Bbox coords relative to full image
            track_ids = results[0].boxes.id.cpu().numpy()
            self.get_logger().info(f'--- Found {len(track_ids)} objects in initial track ---')

            for box, track_id in zip(boxes, track_ids):
                # Calculate the center of this object's box
                box_center_x = (box[0] + box[2]) / 2
                box_center_y = (box[1] + box[3]) / 2

                # --- MODIFICATION: Check if the object's center is INSIDE the user's box ---
                if x1 <= box_center_x <= x2 and y1 <= box_center_y <= y2:
                    # This object is a candidate. Find the one closest to the ROI center.
                    dist = math.sqrt((roi_center_x - box_center_x)**2 + (roi_center_y - box_center_y)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        selected_id = int(track_id)
                        selected_bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

        if selected_id is not None:
            target_track_id = selected_id
            last_known_bbox = selected_bbox
            self.get_logger().info(f'---> Target Selected! Tracking ID: {target_track_id} <---')
        else:
            self.get_logger().warn('--- No trackable object found INSIDE the selected ROI. ---')


    def image_callback(self, msg):
        global roi_start_point, roi_end_point, drawing, roi_defined, target_track_id, last_known_bbox, valid_roi_coords

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_frame_shape = cv_image.shape
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        display_image = cv_image.copy() # Start with a fresh image copy

        # --- Draw Center of POV (Point of View) ---
        h, w = display_image.shape[:2]
        pov_center_x = w // 2
        pov_center_y = h // 2
        cv2.drawMarker(display_image, (pov_center_x, pov_center_y), (255, 255, 255), 
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

        # --- Step 1: Handle ROI definition if requested ---
        if roi_defined:
            self.find_target_id_in_roi(cv_image)
            roi_defined = False # Reset flag after attempting to find ID

        # --- Step 2: Track target object if ID is set ---
        if target_track_id is not None:
            try:
                # Run tracking on the FULL image
                results = self.model.track(cv_image, persist=True, verbose=False)

                found_target = False
                if results and results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy()

                    for box, track_id in zip(boxes, track_ids):
                        if int(track_id) == target_track_id:
                            x1, y1, x2, y2 = map(int, box)
                            last_known_bbox = (x1, y1, x2, y2)
                            
                            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            label = f"ID: {target_track_id}"
                            cv2.putText(display_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            obj_center_x = (x1 + x2) // 2
                            obj_center_y = (y1 + y2) // 2
                            cv2.circle(display_image, (obj_center_x, obj_center_y), 5, (0, 255, 0), -1) 

                            found_target = True
                            break

                if not found_target:
                    self.get_logger().warn(f'--- Track ID {target_track_id} lost! ---')
                    if last_known_bbox:
                         cv2.rectangle(display_image, (last_known_bbox[0], last_known_bbox[1]), (last_known_bbox[2], last_known_bbox[3]), (0, 0, 255), 2)

            except Exception as e:
                self.get_logger().error(f"Error during YOLO tracking: {e}")
                target_track_id = None 
                last_known_bbox = None

        # --- Step 3: Draw temporary ROI selection box ---
        if drawing:
            cv2.rectangle(display_image, roi_start_point, roi_end_point, (0, 255, 255), 2) # YELLOW box while drawing

        # --- Step 4: Display the image ---
        cv2.imshow(self.window_name, display_image)
        key = cv2.waitKey(1)

    def destroy_node(self):
        self.get_logger().info('Destroying node, closing OpenCV window.')
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    yolo_subscriber = None
    try:
        yolo_subscriber = YoloSubscriber()
        rclpy.spin(yolo_subscriber)
    except RuntimeError as e:
        if 'yolo_subscriber' in locals() and yolo_subscriber:
             yolo_subscriber.get_logger().error(f"Node initialization failed: {e}")
        else:
             print(f"Node initialization failed before logger setup: {e}")
    except KeyboardInterrupt:
        if 'yolo_subscriber' in locals() and yolo_subscriber:
             yolo_subscriber.get_logger().info('KeyboardInterrupt, shutting down.')
        else:
             print('KeyboardInterrupt during initialization, shutting down.')
    finally:
        if 'yolo_subscriber' in locals() and yolo_subscriber:
            yolo_subscriber.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()

if __name__ == '__main__':
    main()