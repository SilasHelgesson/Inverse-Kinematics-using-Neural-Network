import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time
import threading

class MoveItIKSolver(Node):
    def __init__(self):
        super().__init__('traditional_ik_solver')
        
        # Wait for MoveIt IK service
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available, waiting...')
        
        self.joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 
                           'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
        self.current_joint_positions = [0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483]
        self.last_ik_solution = [0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483]
        # Joint state publisher
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.033, self.publish_joint_state)
        
        # Action client
        self._action_client = ActionClient(self, FollowJointTrajectory, '/fr3_arm_controller/follow_joint_trajectory')
        
        # Subscribers
        self.target_sub = self.create_subscription(
            PoseStamped, '/target_ee_pose', self.target_callback, 10)
        
        # Motion state
        self.target_positions = self.current_joint_positions.copy()
        self.motion_start_time = 0
        self.motion_duration = 0
        self.is_moving = False
        
        self.get_logger().info("MoveIt IK Solver initialized")
    
    def publish_joint_state(self):
        """Publish current joint state at 30 Hz"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names + ['fr3_finger_joint1', 'fr3_finger_joint2']
        msg.position = self.current_joint_positions + [0.035, 0.035]
        msg.velocity = [0.0] * len(msg.name)
        
        self.joint_state_pub.publish(msg)
    
    def target_callback(self, msg: PoseStamped):
        """Receive target pose and solve IK using MoveIt"""
        self.get_logger().info(f"Target: ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})")
        
        # Call MoveIt IK service
        request = GetPositionIK.Request()
        request.ik_request.group_name = "fr3_arm"
        request.ik_request.pose_stamped = msg
        request.ik_request.timeout.sec = 5
        
        # Use current state as seed
        request.ik_request.robot_state.joint_state.name = self.joint_names
        request.ik_request.robot_state.joint_state.position = self.current_joint_positions
        
        future = self.ik_client.call_async(request)
        future.add_done_callback(self.ik_callback)
        request.ik_request.robot_state.joint_state.position = self.last_ik_solution  # Use last solution, not current
    def ik_callback(self, future):
        """Handle IK response"""
        try:
            response = future.result()
            
            if response.error_code.val != 1:  # Not SUCCESS
                self.get_logger().warn(f"IK failed with code {response.error_code.val}")
                return
            
            joint_angles = list(response.solution.joint_state.position[:7])
            self.get_logger().info(f"IK Solution: {[f'{j:.3f}' for j in joint_angles]}")
            
            # Start motion
            self.target_positions = joint_angles
            self.motion_start_time = time.time()
            self.motion_duration = 5.0
            self.is_moving = True
            self.last_ik_solution = joint_angles
        except Exception as e:
            self.get_logger().error(f"IK callback error: {e}")
    
    def update_motion(self):
        """Update joint positions if in motion"""
        if self.is_moving:
            elapsed = time.time() - self.motion_start_time
            if elapsed < self.motion_duration:
                alpha = elapsed / self.motion_duration
                for i in range(len(self.current_joint_positions)):
                    self.current_joint_positions[i] = (
                        self.current_joint_positions[i] * (1 - alpha) + 
                        self.target_positions[i] * alpha
                    )
            else:
                self.current_joint_positions = self.target_positions.copy()
                self.is_moving = False

def main(args=None):
    rclpy.init(args=args)
    solver = MoveItIKSolver()
    
    # Update motion thread
    def motion_thread():
        while rclpy.ok():
            solver.update_motion()
            time.sleep(0.01)
    
    thread = threading.Thread(target=motion_thread, daemon=True)
    thread.start()
    
    rclpy.spin(solver)
    rclpy.shutdown()

if __name__ == '__main__':
    main()