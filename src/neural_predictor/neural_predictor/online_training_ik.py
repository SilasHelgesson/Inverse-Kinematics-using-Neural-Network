import os
import time

import rclpy

from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK

import torch
import torch.nn as nn

import numpy as np


# FR3 joint limits (rad), used only to clamp what we publish for RViz
FR3_JOINT_LOWER = np.array(
    [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_JOINT_UPPER = np.array(
    [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

# Fixed IK seed. Every label is computed from this same seed,
# so a given target always maps to the same joint configuration.
READY_POSE = [
    0.0,
    -0.7853981633974483,
    0.0,
    -2.356194490192345,
    0.0,
    1.5707963267948966,
    0.7853981633974483,
]

# fr3_joint2 sits about 0.333 m above fr3_link0
SHOULDER = np.array([0.0, 0.0, 0.333])


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
    """Train the IK network on batches of MoveIt IK solutions.

    Training runs in the background, independent of the robot in RViz:
    the node samples `batch_size` random targets, solves them all with
    /compute_ik in parallel, and trains on the results once the batch
    is complete, then starts the next batch.

    Targets on /target_ee_pose are only used for the RViz comparison
    (network prediction vs MoveIt); they are not trained on.
    """

    def __init__(self):
        super().__init__('online_ik_trainer')

        # ---------------- Parameters ----------------
        default_path = os.path.expanduser('~/.ros/ik_model.pth')
        self.declare_parameter('model_path', default_path)

        self.declare_parameter('batch_size', 100)        # IK solutions per batch
        self.declare_parameter('epochs_per_batch', 1)    # gradient steps per batch
        self.declare_parameter('ik_timeout', 0.5)        # s, per IK request
        self.declare_parameter('batch_stale_timeout', 30.0)  # s, give up waiting
        self.declare_parameter('training_enabled', True)

        # Must match the target generator so training covers the same space
        self.declare_parameter('min_reach', 0.30)
        self.declare_parameter('max_reach', 0.70)
        self.declare_parameter('min_radius_xy', 0.25)
        self.declare_parameter('min_z', 0.05)

        self.model_path = os.path.abspath(os.path.expanduser(
            self.get_parameter('model_path').value))
        self.batch_size = int(self.get_parameter('batch_size').value)
        self.epochs_per_batch = int(self.get_parameter('epochs_per_batch').value)
        self.ik_timeout = float(self.get_parameter('ik_timeout').value)
        self.batch_stale_timeout = float(
            self.get_parameter('batch_stale_timeout').value)
        self.training_enabled = bool(
            self.get_parameter('training_enabled').value)

        # ---------------- MoveIt IK service ----------------
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

        # ---------------- Network ----------------
        self.model = IKNetwork()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        self.train_steps = 0
        self.batches_done = 0
        self.loss_value = 0.0
        self.load_model()

        # ---------------- Batch state ----------------
        self.batch_id = 0
        self.batch_pending = 0
        self.batch_inputs = []
        self.batch_labels = []
        self.batch_failures = 0
        self.batch_start_time = None

        # ---------------- Publishers ----------------
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

        # ---------------- Display targets ----------------
        self.target_sub = self.create_subscription(
            PoseStamped,
            '/target_ee_pose',
            self.target_callback,
            10
        )

        # Starts a new batch whenever the previous one has finished
        self.batch_timer = self.create_timer(0.05, self.batch_tick)

        self.get_logger().info(
            f"Online IK Trainer ready (batch_size={self.batch_size}, "
            f"epochs_per_batch={self.epochs_per_batch})")

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def load_model(self):
        if not os.path.exists(self.model_path):
            self.get_logger().info(
                f"No model at {self.model_path}, starting with random weights")
            return

        try:
            checkpoint = torch.load(
                self.model_path, map_location='cpu', weights_only=True)

            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.train_steps = int(checkpoint.get('train_steps', 0))
            else:
                # Old format: a bare state_dict
                self.model.load_state_dict(checkpoint)

            self.get_logger().info(
                f"Loaded model from {self.model_path} "
                f"({self.train_steps} previous training steps)")

        except Exception as e:
            # Keep the broken file for inspection instead of overwriting it
            backup = self.model_path + '.corrupt'
            os.replace(self.model_path, backup)
            self.get_logger().error(
                f"Failed to load {self.model_path}: {e}. "
                f"Moved it to {backup}, starting with random weights")

    def save_model(self):
        """Save atomically: write a temp file, then rename over the old one."""
        tmp_path = self.model_path + '.tmp'
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(
                {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_steps': self.train_steps,
                },
                tmp_path
            )
            os.replace(tmp_path, self.model_path)
        except Exception as e:
            self.get_logger().warn(f"Failed to save model: {e}")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def sample_position(self):
        """Same reachable-shell sampling as random_target_node."""
        min_reach = self.get_parameter('min_reach').value
        max_reach = self.get_parameter('max_reach').value
        min_radius_xy = self.get_parameter('min_radius_xy').value
        min_z = self.get_parameter('min_z').value

        for _ in range(1000):
            p = np.array([
                np.random.uniform(0.0, max_reach),
                np.random.uniform(-max_reach, max_reach),
                np.random.uniform(min_z, SHOULDER[2] + max_reach),
            ])
            reach = np.linalg.norm(p - SHOULDER)
            radius_xy = np.linalg.norm(p[:2])
            if min_reach <= reach <= max_reach and radius_xy >= min_radius_xy:
                return p

        return np.array([0.5, 0.0, 0.4])

    def make_pose(self, position):
        """Pose in fr3_link0 with the gripper pointing down.

        The network only sees position, so this orientation must match
        the one used by random_target_node.
        """
        msg = PoseStamped()
        msg.header.frame_id = 'fr3_link0'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = 1.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 0.0
        return msg

    def make_ik_request(self, pose_stamped):
        request = GetPositionIK.Request()
        request.ik_request.group_name = "fr3_arm"
        request.ik_request.pose_stamped = pose_stamped
        request.ik_request.timeout = Duration(seconds=self.ik_timeout).to_msg()

        # Always the same seed, so labels are consistent per target
        request.ik_request.robot_state.joint_state.name = self.joint_names
        request.ik_request.robot_state.joint_state.position = READY_POSE
        return request

    def extract_joints(self, response):
        """Return the 7 arm joints in order, or None on failure."""
        if response.error_code.val != 1:
            return None

        solution = response.solution.joint_state
        joint_positions = dict(zip(solution.name, solution.position))

        if any(name not in joint_positions for name in self.joint_names):
            return None

        return np.array([joint_positions[name] for name in self.joint_names])

    # ------------------------------------------------------------------
    # Background batch training
    # ------------------------------------------------------------------

    def batch_tick(self):
        if not self.training_enabled:
            return

        if self.batch_start_time is not None:
            # A batch is in flight; train early if some requests never return
            age = time.monotonic() - self.batch_start_time
            if age > self.batch_stale_timeout:
                self.get_logger().warn(
                    f"Batch timed out with {self.batch_pending} requests "
                    f"outstanding, training on what arrived")
                self.finish_batch()
            return

        self.start_batch()

    def start_batch(self):
        self.batch_id += 1
        self.batch_pending = self.batch_size
        self.batch_inputs = []
        self.batch_labels = []
        self.batch_failures = 0
        self.batch_start_time = time.monotonic()

        batch_id = self.batch_id

        for _ in range(self.batch_size):
            position = self.sample_position()
            request = self.make_ik_request(self.make_pose(position))
            future = self.ik_client.call_async(request)
            future.add_done_callback(
                lambda f, p=position: self.batch_ik_callback(f, p, batch_id)
            )

    def batch_ik_callback(self, future, position, batch_id):
        # Ignore late answers from a batch that already timed out
        if batch_id != self.batch_id or self.batch_start_time is None:
            return

        try:
            joints = self.extract_joints(future.result())
        except Exception as e:
            self.get_logger().error(f"Batch IK error: {e}")
            joints = None

        if joints is None:
            self.batch_failures += 1
        else:
            self.batch_inputs.append(position)
            self.batch_labels.append(joints)

        self.batch_pending -= 1
        if self.batch_pending <= 0:
            self.finish_batch()

    def finish_batch(self):
        elapsed = time.monotonic() - self.batch_start_time
        n = len(self.batch_inputs)

        if n > 0:
            self.train_on_batch(
                np.array(self.batch_inputs),
                np.array(self.batch_labels)
            )
            self.batches_done += 1
            self.save_model()

        self.get_logger().info(
            f"Batch {self.batch_id}: {n} solutions, "
            f"{self.batch_failures} IK failures, {elapsed:.2f}s | "
            f"Loss: {self.loss_value:.6f} | Steps: {self.train_steps}"
        )

        # Lets batch_tick start the next one
        self.batch_start_time = None

    def train_on_batch(self, inputs, labels):
        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        for _ in range(self.epochs_per_batch):
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_value = loss.item()
            self.train_steps += 1

    # ------------------------------------------------------------------
    # RViz comparison (display only, no training)
    # ------------------------------------------------------------------

    def target_callback(self, msg: PoseStamped):
        """Predict with the network and get MoveIt's answer for display."""

        target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        with torch.no_grad():
            pred_tensor = self.model(
                torch.tensor(target_pos, dtype=torch.float32))
            network_angles = pred_tensor.cpu().numpy()

        if np.any(np.isnan(network_angles)) or np.any(np.isinf(network_angles)):
            self.get_logger().warn("Network output contains NaN/inf, skipping")
            return

        future = self.ik_client.call_async(self.make_ik_request(msg))
        future.add_done_callback(
            lambda f: self.display_ik_callback(f, network_angles)
        )

    def display_ik_callback(self, future, network_angles):
        try:
            ground_truth_angles = self.extract_joints(future.result())

            if ground_truth_angles is None:
                self.get_logger().warn("MoveIt IK failed for display target")
                return

            self.publish_solutions(network_angles, ground_truth_angles)

            error = np.linalg.norm(network_angles - ground_truth_angles)

            self.get_logger().info(
                f"Display | "
                f"Net: {[f'{j:.2f}' for j in network_angles]} | "
                f"GT: {[f'{j:.2f}' for j in ground_truth_angles]} | "
                f"Err: {error:.4f}"
            )

        except Exception as e:
            self.get_logger().error(f"Display IK callback error: {e}")

    def publish_solutions(self, network_angles, ground_truth_angles):
        """Publish both solutions for RViz comparison."""

        # Clamp so RViz never shows physically impossible poses
        network_angles = np.clip(
            network_angles, FR3_JOINT_LOWER, FR3_JOINT_UPPER)

        names = self.joint_names + ['fr3_finger_joint1', 'fr3_finger_joint2']
        stamp = self.get_clock().now().to_msg()

        msg1 = JointState()
        msg1.header.stamp = stamp
        msg1.header.frame_id = 'base'
        msg1.name = names
        msg1.position = [float(a) for a in network_angles] + [0.035, 0.035]
        msg1.velocity = [0.0] * len(names)
        self.network_joint_pub.publish(msg1)

        # MoveIt ground-truth solution (for rosbag / plotting)
        msg2 = JointState()
        msg2.header.stamp = stamp
        msg2.header.frame_id = 'base'
        msg2.name = names
        msg2.position = [float(a) for a in ground_truth_angles] + [0.035, 0.035]
        msg2.velocity = [0.0] * len(names)
        self.ground_truth_joint_pub.publish(msg2)


def main(args=None):

    rclpy.init(args=args)

    trainer = OnlineIKTrainer()

    try:
        rclpy.spin(trainer)
    except KeyboardInterrupt:
        pass
    finally:
        trainer.save_model()
        trainer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
