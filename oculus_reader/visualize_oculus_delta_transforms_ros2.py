from reader import OculusReader
from tf_transformations import quaternion_from_matrix
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension
import numpy as np

# OpenXR (Oculus, Y-up / -Z-forward) → ROS REP-103 (Z-up / X-forward)
# x_ros = -z_oculus, y_ros = -x_oculus, z_ros = +y_oculus
_M_TO_ROS = np.array([
    [ 0,  0, -1, 0],
    [-1,  0,  0, 0],
    [ 0,  1,  0, 0],
    [ 0,  0,  0, 1],
], dtype=float)


def _to_ros_frame(transform: np.ndarray) -> np.ndarray:
    """Change-of-basis: OpenXR 4×4 homogeneous matrix → ROS frame."""
    return _M_TO_ROS @ transform @ _M_TO_ROS.T


class OculusDeltaReaderNode(Node):
    def __init__(self):
        super().__init__('oculus_delta_reader')
        self.oculus_reader = OculusReader()
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.timer_callback)

        # Anchor poses captured on trigger press, per controller.
        # None means the trigger is currently released.
        self.anchors = {'r': None, 'l': None}
        self.trigger_keys = {'r': 'RG', 'l': 'LG'}
        self.child_frames = {'r': 'oculus_r_delta', 'l': 'oculus_l_delta'}
        self.matrix_publishers = {
            'r': self.create_publisher(Float64MultiArray, 'oculus/r_delta_matrix', 10),
            'l': self.create_publisher(Float64MultiArray, 'oculus/l_delta_matrix', 10),
        }
        self.trigger_publishers = {
            'r': self.create_publisher(Bool, 'oculus/r_move_enable', 10),
            'l': self.create_publisher(Bool, 'oculus/l_move_enable', 10),
        }

        # Jump filter: if a new delta differs from the last accepted one by
        # more than these per-tick thresholds, hold the previous delta until
        # the controller re-enters the safe boundary.
        self.last_delta = {'r': None, 'l': None}
        self.max_trans_step = 0.1   # meters per tick (~2 m/s at 20 Hz)
        self.max_rot_step = 0.5     # radians per tick (~10 rad/s at 20 Hz)

    def timer_callback(self):
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        if not transformations:
            return

        # Capture the data-read time once per tick; used as the message timestamp so
        # subscribers can compute accurate dt without relying on callback-execution time.
        stamp_ns = float(self.get_clock().now().nanoseconds)

        for side in ('r', 'l'):
            if side not in transformations:
                continue
            pose = _to_ros_frame(transformations[side])
            pressed = bool(buttons.get(self.trigger_keys[side], False))
            self.trigger_publishers[side].publish(Bool(data=pressed))

            if pressed:
                if self.anchors[side] is None:
                    # Rising edge: capture anchor pose.
                    self.anchors[side] = pose.copy()
                    self.last_delta[side] = np.eye(4)
                    self.get_logger().info(f'[{side}] trigger pressed — anchor captured')
                # Delta expressed in the anchor's local frame:
                #   delta = inv(anchor) @ current
                # so the published frame starts at identity and moves as the
                # controller diverges from the anchor pose.
                delta = np.linalg.inv(self.anchors[side]) @ pose
                prev = self.last_delta[side]
                if prev is not None and self._delta_jump(prev, delta):
                    # Hold the previous delta until the new one is back within
                    # the safe per-tick boundary.
                    delta = prev
                self.last_delta[side] = delta
                self.publish_transform(delta, self.child_frames[side])
                self.publish_matrix(delta, side, stamp_ns)
            else:
                if self.anchors[side] is not None:
                    self.get_logger().info(f'[{side}] trigger released — delta reset')
                    self.anchors[side] = None
                    self.last_delta[side] = None
                # While disengaged, publish identity so RViz still shows the frame.
                identity = np.eye(4)
                self.publish_transform(identity, self.child_frames[side])
                self.publish_matrix(identity, side, stamp_ns)

    def publish_transform(self, transform, name):
        translation = transform[:3, 3]
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = name
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        quat = quaternion_from_matrix(transform)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.br.sendTransform(t)

    def _delta_jump(self, prev, current):
        trans_step = float(np.linalg.norm(current[:3, 3] - prev[:3, 3]))
        rel_rot = prev[:3, :3].T @ current[:3, :3]
        cos_angle = np.clip((np.trace(rel_rot) - 1.0) * 0.5, -1.0, 1.0)
        rot_step = float(np.arccos(cos_angle))
        if trans_step > self.max_trans_step or rot_step > self.max_rot_step:
            self.get_logger().warn(
                f'delta jump filtered: Δt={trans_step:.3f} m, Δθ={rot_step:.3f} rad'
            )
            return True
        return False

    def publish_matrix(self, transform, side, stamp_ns: float) -> None:
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='data', size=17, stride=17),
        ]
        flat = np.asarray(transform, dtype=np.float64).reshape(-1).tolist()
        flat.append(float(stamp_ns))  # element [16]: nanoseconds since epoch (float64)
        msg.data = flat
        self.matrix_publishers[side].publish(msg)


def main():
    rclpy.init()
    node = OculusDeltaReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()