import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import uuid

class MarkerPublisher(Node):
    def __init__(self):
        super().__init__('marker_publisher')
        
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.target_sub = self.create_subscription(
            PoseStamped, '/target_ee_pose', self.target_callback, 10)
    
    def target_callback(self, msg: PoseStamped):
        """Publish target position as red sphere"""
        marker = Marker()
        marker.header = msg.header
        marker.ns = "target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose = msg.pose
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # Red color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    publisher = MarkerPublisher()
    rclpy.spin(publisher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()