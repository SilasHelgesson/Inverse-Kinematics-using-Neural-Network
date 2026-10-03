import rclpy

from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK

import torch
import torch.nn as nn

import numpy as np


class IKNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        )

    def forward(self, x):
        return self.net(x)


class OnlineIKTrainer(Node):

    def __init__(self):
        super().__init__('online_ik_trainer')

        # MoveIt IK service
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for IK service...')

        self.joint_names = [
            'fr3_joint1',
            'fr3_joint2',
            'fr3_joint3',
            'fr3_joint4',
            'fr3_joint5',
            'fr3_joint6',
            'fr3_joint7'
        ]

        # Network
        self.model = IKNetwork()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        try:
            self.model.load_state_dict(torch.load('ik_model.pth'))
            self.get_logger().info("Loaded pretrained model")
        except Exception:
            self.get_logger().info("Starting with random weights")

        self.current_joints = [
            0.0,
            -0.7853981633974483,
            0.0,
            -2.356194490192345,
            0.0,
            1.5707963267948966,
            0.7853981633974483
        ]

        # Publishers
        self.network_joint_pub = self.create_publisher(
            JointState,
            '/network_joints',
            10
        )

        self.ground_truth_joint_pub = self.create_publisher(
            JointState,
            '/ground_truth_joints',
            10
        )

        # Target subscriber
        self.target_sub = self.create_subscription(
            PoseStamped,
            '/target_ee_pose',
            self.target_callback,
            10
        )

        self.loss_value = 0.0

        self.get_logger().info("Online IK Trainer ready")

    def target_callback(self, msg: PoseStamped):
        """Get target, predict with network, call MoveIt async."""

        target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        # Network prediction
        with torch.no_grad():
            target_tensor = torch.tensor(
                target_pos,
                dtype=torch.float32
            )

            pred_tensor = self.model(target_tensor)
            network_angles = pred_tensor.detach().cpu().numpy()
            
            # Validate network output
            if np.any(np.isnan(network_angles)) or np.any(np.isinf(network_angles)):
                self.get_logger().warn("Network output contains NaN/inf, skipping this iteration")
                return

        # MoveIt IK request
        request = GetPositionIK.Request()

        request.ik_request.group_name = "fr3_arm"
        request.ik_request.pose_stamped = msg

        request.ik_request.timeout.sec = 2

        request.ik_request.robot_state.joint_state.name = self.joint_names
        request.ik_request.robot_state.joint_state.position = self.current_joints

        # Call async
        future = self.ik_client.call_async(request)

        future.add_done_callback(
            lambda f: self.ik_callback(
                f,
                target_pos,
                network_angles
            )
        )

    def ik_callback(self, future, target_pos, network_angles):
        """Handle IK response when it comes back."""

        try:
            response = future.result()

            if response.error_code.val != 1:
                self.get_logger().warn(
                    f"MoveIt IK failed: {response.error_code.val}"
                )
                return

            # ---------------------------------------------------------
            # IMPORTANT:
            # MoveIt does not guarantee that joint_state.position[:7]
            # corresponds to fr3_joint1 ... fr3_joint7.
            #
            # Match each position to its joint name explicitly.
            # ---------------------------------------------------------
            solution = response.solution.joint_state

            joint_positions = dict(
                zip(solution.name, solution.position)
            )

            missing_joints = [
                name
                for name in self.joint_names
                if name not in joint_positions
            ]

            if missing_joints:
                self.get_logger().error(
                    f"MoveIt solution is missing joints: {missing_joints}"
                )
                return

            ground_truth_angles = np.array([
                joint_positions[name]
                for name in self.joint_names
            ])

            # Train network
            self.train_step(
                target_pos,
                ground_truth_angles
            )

            # Update current configuration
            self.current_joints = ground_truth_angles.tolist()

            # Publish NN and ground-truth solutions
            self.publish_solutions(
                network_angles,
                ground_truth_angles
            )

            # Compare NN prediction against MoveIt solution
            error = np.linalg.norm(
                network_angles - ground_truth_angles
            )

            self.get_logger().info(
                f"Net: {[f'{j:.2f}' for j in network_angles]} | "
                f"GT: {[f'{j:.2f}' for j in ground_truth_angles]} | "
                f"Err: {error:.4f} | "
                f"Loss: {self.loss_value:.6f}"
            )

        except Exception as e:
            self.get_logger().error(
                f"IK callback error: {e}"
            )

    def train_step(self, target, ground_truth):
        """Train network on single sample."""

        target_tensor = torch.tensor(
            target,
            dtype=torch.float32
        )

        ground_truth_tensor = torch.tensor(
            ground_truth,
            dtype=torch.float32
        )

        pred = self.model(target_tensor)

        loss = self.loss_fn(
            pred,
            ground_truth_tensor
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.loss_value = loss.item()

        # Occasionally save the model
        if np.random.random() < 0.1:
            torch.save(
                self.model.state_dict(),
                'ik_model.pth'
            )

    def publish_solutions(
        self,
        network_angles,
        ground_truth_angles
    ):
        """Publish both solutions for RViz comparison."""

        # Neural network solution
        msg1 = JointState()

        msg1.header.stamp = self.get_clock().now().to_msg()
        msg1.header.frame_id = 'base'

        msg1.name = self.joint_names + [
            'fr3_finger_joint1',
            'fr3_finger_joint2'
        ]

        msg1.position = list(network_angles) + [
            0.035,
            0.035
        ]
        msg1.velocity = [0.0] * len(msg1.name)

        self.network_joint_pub.publish(msg1)

        # MoveIt ground-truth solution
        msg2 = JointState()

        msg2.header.stamp = self.get_clock().now().to_msg()
        msg2.header.frame_id = 'base'

        msg2.name = self.joint_names + [
            'fr3_finger_joint1',
            'fr3_finger_joint2'
        ]

        msg2.position = list(ground_truth_angles) + [
            0.035,
            0.035
        ]
        msg2.velocity = [0.0] * len(msg2.name)

        self.ground_truth_joint_pub.publish(msg2)


def main(args=None):

    rclpy.init(args=args)

    trainer = OnlineIKTrainer()

    rclpy.spin(trainer)

    rclpy.shutdown()


if __name__ == '__main__':
    main()