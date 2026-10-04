import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class MoveItIKSolver(Node):
    """Solve IK with MoveIt and send the result to the arm controller.

    This node no longer publishes /joint_states. The controller's
    joint_state_broadcaster owns that topic; we only read from it.
    """

    def __init__(self):
        super().__init__('traditional_ik_solver')

        self.joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7',
        ]

        # Fallback seed until the first /joint_states message arrives
        self.current_joint_positions = [
            0.0, -0.7853981633974483, 0.0, -2.356194490192345,
            0.0, 1.5707963267948966, 0.7853981633974483,
        ]

        # Max joint speed used to pick a trajectory duration (rad/s)
        self.max_joint_speed = 0.5
        self.min_duration = 1.0

        # MoveIt IK service
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available, waiting...')

        # Arm controller
        self._action_client = ActionClient(
            self, FollowJointTrajectory,
            '/fr3_arm_controller/follow_joint_trajectory')
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Arm controller not available, waiting...')

        # Read the real robot state instead of publishing our own
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.target_sub = self.create_subscription(
            PoseStamped, '/target_ee_pose', self.target_callback, 10)

        self.get_logger().info('MoveIt IK Solver initialized')

    def joint_state_callback(self, msg: JointState):
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in self.joint_names):
            self.current_joint_positions = [
                positions[name] for name in self.joint_names
            ]

    def target_callback(self, msg: PoseStamped):
        p = msg.pose.position
        self.get_logger().info(f'Target: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')

        request = GetPositionIK.Request()
        request.ik_request.group_name = 'fr3_arm'
        request.ik_request.pose_stamped = msg
        request.ik_request.timeout.sec = 5

        # Seed with where the arm actually is right now
        request.ik_request.robot_state.joint_state.name = self.joint_names
        request.ik_request.robot_state.joint_state.position = list(
            self.current_joint_positions)

        future = self.ik_client.call_async(request)
        future.add_done_callback(self.ik_callback)

    def ik_callback(self, future):
        try:
            response = future.result()

            if response.error_code.val != 1:
                self.get_logger().warn(
                    f'IK failed with code {response.error_code.val}')
                return

            # Pick joints by name; the solution may contain extra joints
            # (e.g. fingers) in any order
            solution = response.solution.joint_state
            positions = dict(zip(solution.name, solution.position))
            missing = [n for n in self.joint_names if n not in positions]
            if missing:
                self.get_logger().error(f'IK solution missing joints: {missing}')
                return
            joint_angles = [positions[n] for n in self.joint_names]

            self.get_logger().info(
                f"IK Solution: {[f'{j:.3f}' for j in joint_angles]}")

            self.send_trajectory(joint_angles)

        except Exception as e:
            self.get_logger().error(f'IK callback error: {e}')

    def send_trajectory(self, joint_angles):
        # Scale duration by the largest joint move so speed stays sane
        max_delta = max(
            abs(t - c) for t, c in zip(joint_angles, self.current_joint_positions)
        )
        duration = max(self.min_duration, max_delta / self.max_joint_speed)

        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.velocities = [0.0] * len(joint_angles)
        point.time_from_start = Duration(seconds=duration).to_msg()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points = [point]

        # A new goal preempts the previous one, so the arm blends
        # from wherever it currently is
        self._action_client.send_goal_async(goal)


def main(args=None):
    rclpy.init(args=args)
    solver = MoveItIKSolver()
    rclpy.spin(solver)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
