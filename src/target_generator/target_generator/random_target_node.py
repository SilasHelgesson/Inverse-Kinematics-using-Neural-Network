import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np


class RandomTargetGenerator(Node):
    """Publish random targets that the FR3 can actually reach.

    Positions are rejection-sampled from a box and kept only if they lie
    in a shell around the shoulder joint. The gripper points straight
    down, which is the most reachable fixed orientation for the FR3.
    """

    # fr3_joint2 sits about 0.333 m above fr3_link0
    SHOULDER = np.array([0.0, 0.0, 0.333])

    def __init__(self):
        super().__init__('target_generator')

        self.declare_parameter('period', 5.0)
        self.declare_parameter('min_reach', 0.30)   # from shoulder (m)
        self.declare_parameter('max_reach', 0.70)   # FR3 max is ~0.855 m
        self.declare_parameter('min_radius_xy', 0.25)  # stay off the base
        self.declare_parameter('min_z', 0.05)

        self.pub = self.create_publisher(PoseStamped, '/target_ee_pose', 10)
        period = float(self.get_parameter('period').value)
        self.timer = self.create_timer(period, self.publish_target)

    def sample_position(self):
        min_reach = self.get_parameter('min_reach').value
        max_reach = self.get_parameter('max_reach').value
        min_radius_xy = self.get_parameter('min_radius_xy').value
        min_z = self.get_parameter('min_z').value

        for _ in range(1000):
            p = np.array([
                np.random.uniform(0.0, max_reach),
                np.random.uniform(-max_reach, max_reach),
                np.random.uniform(min_z, self.SHOULDER[2] + max_reach),
            ])
            reach = np.linalg.norm(p - self.SHOULDER)
            radius_xy = np.linalg.norm(p[:2])
            if min_reach <= reach <= max_reach and radius_xy >= min_radius_xy:
                return p

        # Should never happen with sane parameters
        self.get_logger().warn('Sampling failed, using fallback target')
        return np.array([0.5, 0.0, 0.4])

    def publish_target(self):
        msg = PoseStamped()
        msg.header.frame_id = 'fr3_link0'
        msg.header.stamp = self.get_clock().now().to_msg()

        x, y, z = self.sample_position()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        # Fixed orientation: gripper pointing straight down (unit quaternion)
        msg.pose.orientation.x = 1.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 0.0

        self.pub.publish(msg)
        self.get_logger().info(f"Target: ({x:.2f}, {y:.2f}, {z:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = RandomTargetGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
