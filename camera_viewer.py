import argparse
import cv2
import numpy as np
import platform
import sys
import time
from pathlib import Path
from typing import Optional


class SimpleCameraViewer:
    """Simple Camera Viewer Class"""
    
    def __init__(self, camera_index: int = 0, fps: Optional[int] = None, 
                 width: Optional[int] = None, height: Optional[int] = None,
                 color_mode: str = "bgr", background_image: Optional[str] = None,
                 background_alpha: float = 0.5):
        """
        Initialize the Camera Viewer
        
        Args:
            camera_index: Camera index, default is 0
            fps: Frame rate, None means use default value
            width: Image width, None means use default value
            height: Image height, None means use default value
            color_mode: Color mode, "rgb" or "bgr"
            background_image: Path to background image file
            background_alpha: Background image transparency (0.0-1.0)
        """
        self.camera_index = camera_index
        self.fps = fps
        self.width = width
        self.height = height
        self.color_mode = color_mode
        self.background_image_path = background_image
        self.background_alpha = background_alpha
        
        self.camera = None
        self.is_connected = False
        self.background_image = None
        
        # validate color mode
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(f"Color mode must be 'rgb' or 'bgr', but provided {self.color_mode}")
        
        # validate background alpha
        if not 0.0 <= self.background_alpha <= 1.0:
            raise ValueError(f"Background alpha must be between 0.0 and 1.0, but provided {self.background_alpha}")
        
        # load background image
        if self.background_image_path:
            self._load_background_image()
    
    def _load_background_image(self):
        """Load and prepare background image"""
        if not self.background_image_path:
            return
        
        if not Path(self.background_image_path).exists():
            raise FileNotFoundError(f"Background image not found: {self.background_image_path}")
        
        # load background image
        self.background_image = cv2.imread(self.background_image_path)
        if self.background_image is None:
            raise ValueError(f"Cannot load background image: {self.background_image_path}")
        
        print(f"Background image loaded: {self.background_image_path}")
        print(f"  Size: {self.background_image.shape[1]} x {self.background_image.shape[0]}")
    
    def _resize_background_to_match_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize background image to match frame size"""
        if self.background_image is None:
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        bg_height, bg_width = self.background_image.shape[:2]
        
        # if background image size does not match frame, resize it
        if bg_width != frame_width or bg_height != frame_height:
            resized_bg = cv2.resize(self.background_image, (frame_width, frame_height))
        else:
            resized_bg = self.background_image.copy()
        
        return resized_bg
    
    def _overlay_background(self, frame: np.ndarray) -> np.ndarray:
        """Overlay background image on the frame"""
        if self.background_image is None:
            return frame
        
        # resize background image to match frame
        resized_bg = self._resize_background_to_match_frame(frame)
        
        # blend images
        # use weighted average for blending: result = alpha * background + (1 - alpha) * frame
        blended_frame = cv2.addWeighted(resized_bg, self.background_alpha, 
                                       frame, 1.0 - self.background_alpha, 0)
        
        return blended_frame
    
    def find_available_cameras(self) -> list[dict]:
        """Find available cameras"""
        cameras = []
        max_index = 60  # Maximum search index
        
        if platform.system() == "Linux":
            print("Detected Linux system, scanning '/dev/video*' ports for available cameras")
            possible_ports = [str(port) for port in Path("/dev").glob("video*")]
            for port in possible_ports:
                camera = cv2.VideoCapture(port)
                if camera.isOpened():
                    cameras.append({
                        "port": port,
                        "index": int(port.removeprefix("/dev/video"))
                    })
                    print(f"Found camera: {port}")
                camera.release()
        else:
            print(f"Detected {platform.system()} system, scanning index 0 to {max_index}")
            for index in range(max_index):
                camera = cv2.VideoCapture(index)
                if camera.isOpened():
                    cameras.append({
                        "port": None,
                        "index": index
                    })
                    print(f"Found camera index: {index}")
                camera.release()
        
        return cameras
    
    def connect(self):
        """Connect to the camera"""
        if self.is_connected:
            print(f"Camera {self.camera_index} is already connected")
            return
        
        # Select backend
        if platform.system() == "Linux":
            backend = cv2.CAP_V4L2
            camera_idx = f"/dev/video{self.camera_index}"
        elif platform.system() == "Windows":
            backend = cv2.CAP_DSHOW
            camera_idx = self.camera_index
        elif platform.system() == "Darwin":
            backend = cv2.CAP_AVFOUNDATION
            camera_idx = self.camera_index
        else:
            backend = cv2.CAP_ANY
            camera_idx = self.camera_index
        
        # Create camera object
        self.camera = cv2.VideoCapture(camera_idx, backend)
        
        if not self.camera.isOpened():
            # If connection fails, display available cameras
            available_cameras = self.find_available_cameras()
            if not available_cameras:
                raise OSError("No available cameras found")
            
            available_indices = [cam["index"] for cam in available_cameras]
            raise ValueError(
                f"Cannot connect to camera index {self.camera_index}."
                f"Available camera indices: {available_indices}"
            )
        
        # Set camera parameters
        if self.fps is not None:
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Get actual values
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"Camera connected successfully!")
        print(f"  Index: {self.camera_index}")
        print(f"  Actual FPS: {actual_fps:.1f} FPS")
        print(f"  Actual resolution: {actual_width:.0f} x {actual_height:.0f}")
        print(f"  Color mode: {self.color_mode}")
        
        self.is_connected = True
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame"""
        if not self.is_connected:
            raise RuntimeError("Camera is not connected, please call connect() method first")
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Convert color mode - Note: OpenCV imshow requires BGR format
        if self.color_mode == "rgb":
            # If user wants RGB format, we need to convert back to BGR for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # But for display, we need to convert back to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # overlay background image
        frame = self._overlay_background(frame)
        
        return frame
    
    def start_viewer(self, window_name: str = "viewer"):
        """Start the real-time viewer"""
        if not self.is_connected:
            self.connect()
        
        print(f"\nStarting real-time viewer...")
        print("Press 'q' to exit, press 's' to save current frame")
        if self.background_image_path:
            print(f"Background overlay enabled: {self.background_image_path} (alpha: {self.background_alpha})")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    print("Cannot read camera frame")
                    break
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    current_fps = frame_count / elapsed_time
                else:
                    current_fps = 0
                
                # Display information on the image
                info_text = f"FPS: {current_fps:.1f} | frame_count: {frame_count}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # if background image is enabled, display the information
                if self.background_image_path:
                    bg_info = f"BG: {Path(self.background_image_path).name} (alpha={self.background_alpha:.2f})"
                    cv2.putText(frame, bg_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 0), 2)
                
                # Display the image
                cv2.imshow(window_name, frame)
                
                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User pressed 'q', exiting viewer")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"camera_frame_{timestamp}.png"
                    cv2.imwrite(filename, frame)
                    print(f"Image saved: {filename}")
                elif key == ord('+') or key == ord('='):
                    # increase background alpha
                    self.background_alpha = min(1.0, self.background_alpha + 0.1)
                    print(f"Background alpha: {self.background_alpha:.2f}")
                elif key == ord('-'):
                    # decrease background alpha
                    self.background_alpha = max(0.0, self.background_alpha - 0.1)
                    print(f"Background alpha: {self.background_alpha:.2f}")
                
                # Control frame rate
                if self.fps is not None:
                    time.sleep(1.0 / self.fps)
        
        except KeyboardInterrupt:
            print("\nUser interrupted program")
        
        finally:
            self.disconnect()
            cv2.destroyAllWindows()
    
    def disconnect(self):
        """Disconnect the camera"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.is_connected = False
            print("Camera disconnected")
    
    def __del__(self):
        """Destructor, ensure the camera is released correctly"""
        self.disconnect()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Real-time camera viewer")
    parser.add_argument(
        "--camera-index", "-i", 
        type=int, 
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=int,
        default=None,
        help="Set frame rate (default: use camera default value)"
    )
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=None,
        help="Set image width (default: use camera default value)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=None,
        help="Set image height (default: use camera default value)"
    )
    parser.add_argument(
        "--color-mode", "-c",
        choices=["rgb", "bgr"],
        default="bgr",
        help="Color mode (default: bgr)"
    )
    parser.add_argument(
        "--background-image", "-b",
        type=str,
        default=None,
        help="Path to background image for overlay"
    )
    parser.add_argument(
        "--background-alpha", "-a",
        type=float,
        default=0.5,
        help="Background image transparency (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--list-cameras", "-l",
        action="store_true",
        help="List all available cameras"
    )
    
    args = parser.parse_args()
    
    try:
        viewer = SimpleCameraViewer(
            camera_index=args.camera_index,
            fps=args.fps,
            width=args.width,
            height=args.height,
            color_mode=args.color_mode,
            background_image=args.background_image,
            background_alpha=args.background_alpha
        )
        
        if args.list_cameras:
            print("Searching for available cameras...")
            cameras = viewer.find_available_cameras()
            if cameras:
                print(f"\nFound {len(cameras)} cameras:")
                for cam in cameras:
                    print(f"  Index: {cam['index']}, Port: {cam['port']}")
            else:
                print("No cameras found")
            return
        
        # Start the viewer
        viewer.start_viewer()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

