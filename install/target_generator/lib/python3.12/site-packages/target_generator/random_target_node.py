import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np

class RandomTargetGenerator(Node):
    def __init__(self):
        super().__init__('target_generator')
        self.pub = self.create_publisher(PoseStamped, '/target_ee_pose', 10)
        self.timer = self.create_timer(2.0, self.publish_target)
    
    def publish_target(self):
        """Generate random ee_pose around robot workspace"""
        msg = PoseStamped()
        msg.header.frame_id = 'fr3_link0'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Random position (reachable workspace for Franka)
        msg.pose.position.x = np.random.uniform(0.2, 0.8)
        msg.pose.position.y = np.random.uniform(-0.5, 0.5)
        msg.pose.position.z = np.random.uniform(0.1, 0.8)
        
        # Random orientation (quaternion)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.707
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 0.707
        
        self.pub.publish(msg)
        self.get_logger().info(f"Target: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})")

def main(args=None):
    rclpy.init(args=args)
    node = RandomTargetGenerator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()