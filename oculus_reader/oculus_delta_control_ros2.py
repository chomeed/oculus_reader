from reader import OculusReader
from tf_transformations import quaternion_from_matrix
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, Float64MultiArray, Int32, MultiArrayDimension
import numpy as np


def vec_to_reorder_mat(vec):
    """Build a 4x4 axis-reorder matrix from a signed-index vector.

    Mirrors the helper used by VRPolicy in
    oculus_controller_anubis_bimanual_with_robot_pos_with_ros_for_homing.py
    so the delta frame here matches the one consumed by the robot controller.
    """
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


# Axis alignment matching VRPolicy's rmat_reorder = [-2, -1, -3, 4]:
#   x_aligned = -y_oculus
#   y_aligned = -x_oculus
#   z_aligned = -z_oculus
# This is the same convention the bimanual VR controller feeds to the robot,
# so the visualized delta frames here align with what the controller commands.
_RMAT_REORDER = [-2, -1, -3, 4]
_M_ALIGN = vec_to_reorder_mat(_RMAT_REORDER)
_M_ALIGN_INV = np.linalg.inv(_M_ALIGN)


def _to_aligned_frame(transform: np.ndarray) -> np.ndarray:
    """Express an Oculus pose in the aligned (controller-consumed) frame.

    Uses a similarity transform M @ P @ M^-1 so both translation and the body
    axes are remapped, giving RViz a frame whose x/y/z point the same way as
    the robot's control frame.
    """
    return _M_ALIGN @ transform @ _M_ALIGN_INV


class OculusDeltaControlNode(Node):
    def __init__(self, sides=('r', 'l'), grasp_mode: str = 'do_grasp',
                 gripper_only_sides=()):
        super().__init__('oculus_delta_control')
        self.oculus_reader = OculusReader()
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.timer_callback)
        # Sides this node publishes for. Use ('r',) or ('l',) to share the
        # topic namespace with a second publisher (e.g., the keyboard sim).
        self.sides = tuple(sides)

        # Sides whose ARM is disabled (--gripper-only); empty by default, i.e.
        # both arms fully enabled. For such a side we still publish both
        # topics every tick, but pinned to the disengaged values:
        # move_enable=False and an identity delta. Publishing False beats
        # going silent — oculus_servo_node only clears its enable flag on a
        # message, so a side that stops publishing keeps whatever it had
        # last, which can be a stuck True. With False held, that arm's servo
        # stays paused no matter what the hand does.
        #
        # The gripper is deliberately NOT disabled: _handle_grasp gates on
        # the G BUTTON read straight from `buttons`, not on the published
        # move_enable, so squeezing G on a gripper-only side still enables
        # its trigger while the arm stays put.
        self.gripper_only_sides = tuple(gripper_only_sides)
        for side in self.gripper_only_sides:
            self.get_logger().warn(
                f'[{side}] ARM DISABLED (--gripper-only) — move_enable pinned '
                f'False, delta pinned identity; gripper still active.')

        # G trigger: capture anchor → publish delta from anchor.
        # None means the trigger is currently released.
        self.anchors = {'r': None, 'l': None}
        self.trigger_keys = {'r': 'RG', 'l': 'LG'}
        # Joystick click (LJ/RJ): re-set the orientation reference, mirroring
        # VRPolicy.reset_orientation in oculus_controller_anubis_bimanual_*.py.
        self.joystick_keys = {'r': 'RJ', 'l': 'LJ'}
        self.child_frames = {'r': 'oculus_r_delta', 'l': 'oculus_l_delta'}
        self.matrix_publishers = {
            side: self.create_publisher(Float64MultiArray, f'oculus/{side}_delta_matrix', 10)
            for side in self.sides
        }
        self.trigger_publishers = {
            side: self.create_publisher(Bool, f'oculus/{side}_move_enable', 10)
            for side in self.sides
        }

        # Index trigger (LTr/RTr) → grasp signal. Two output formats are
        # supported; pick at launch with --grasp-mode:
        #   'do_grasp' (default) → Bool on /dg_<side>/do_grasp with
        #       TRANSIENT_LOCAL (latched) QoS. Consumed by
        #       no_teleop_grasp_control.py; matches the topic that
        #       robot_env_rl.py publishes during online RL, so the same
        #       grasp consumer works in both teleop and RL.
        #   'pose_id'           → Int32 on /<side>/gripper_pose_id
        #       (0 = open, 2 = close), matching the manus-glove pose-mapper
        #       convention consumed by glove_grasp_control.py.
        #   'both'              → publish both for mixed setups.
        # Published on rising/falling edges only so we don't spam the bus,
        # and only while the side's move-enable (G) is held — see
        # _handle_grasp for the hold-state-on-disengage semantics.
        self.grasp_mode = grasp_mode
        self.index_trigger_keys = {'r': 'RTr', 'l': 'LTr'}
        side_name = {'r': 'right', 'l': 'left'}
        ns_name   = {'r': 'dg_right', 'l': 'dg_left'}
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.gripper_pose_publishers = {}
        self.do_grasp_publishers = {}
        if grasp_mode in ('pose_id', 'both'):
            self.gripper_pose_publishers = {
                side: self.create_publisher(
                    Int32, f'/{side_name[side]}/gripper_pose_id', 10)
                for side in self.sides
            }
        if grasp_mode in ('do_grasp', 'both'):
            self.do_grasp_publishers = {
                side: self.create_publisher(
                    Bool, f'/{ns_name[side]}/do_grasp', latched_qos)
                for side in self.sides
            }
        self.prev_grasp_pressed = {'r': False, 'l': False}

        # Reference-frame calibration (= VRPolicy.vr_to_global_mat). While
        # reset_orientation is True, V continuously re-captures inv(raw) so
        # the controller sits at identity in the visualization. The first G
        # or J press locks V; pressing J again re-arms the tracker.
        self.vr_to_global_mat = {'r': np.eye(4), 'l': np.eye(4)}
        self.reset_orientation = {'r': True, 'l': True}
        # J edge detection so that, when V changes mid-engagement, we also
        # re-capture the G anchor — keeps the published delta continuous.
        self.prev_j_pressed = {'r': False, 'l': False}

        # Jump filter: if a new delta differs from the last accepted one by
        # more than these per-tick thresholds, hold the previous delta until
        # the controller re-enters the safe boundary. Loose on purpose: only
        # true tracking teleports should trip it — fast-but-human hand motion
        # must pass, since the servo side already clips velocities to
        # max_lin_vel / max_ang_vel (0.5 m/s, 1.5 rad/s) before the robot.
        self.last_delta = {'r': None, 'l': None}
        self.max_trans_step = 0.25   # meters per tick (~5 m/s at 20 Hz)
        self.max_rot_step = 1.0      # radians per tick (~20 rad/s at 20 Hz)

    def timer_callback(self):
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()

        # Publish the move-enable (G) Bool every tick for every configured
        # side, regardless of whether `transformations` came back populated.
        # The env-server's `_on_vr_enable` callback is the only thing that
        # ever clears `_vr_enable_*`, so if we skip publishing during a
        # tracking dropout the env-server keeps the last value forever —
        # producing the stuck-True we saw in
        # td3_residual_with_vr_fixed4/buffer.npz, where the actor logged
        # exactly one False→True edge then no further edges across 544
        # supposedly-mixed frames. `buttons` is usually still populated
        # even when transformations are noisy; if it's also empty we
        # default to False (safer: a missed press is a non-issue, a
        # missed release means the robot keeps tracking a human who has
        # let go).
        buttons_map = buttons or {}
        for side in self.sides:
            g_pressed = bool(buttons_map.get(self.trigger_keys[side], False))
            # An arm-disabled side reports move_enable=False forever; the raw
            # button is still passed to _handle_grasp so its gripper works.
            if side in self.gripper_only_sides:
                self.trigger_publishers[side].publish(Bool(data=False))
            else:
                self.trigger_publishers[side].publish(Bool(data=g_pressed))
            # Grasp edges are handled here, next to the move-enable publish,
            # for the same reason: `buttons` survives tracking dropouts that
            # empty `transformations`. Tracking the trigger only in the pose
            # loop below would let a toggle made during a dropout go
            # unrecorded, so the stale prev would fire a phantom edge on the
            # next good frame.
            self._handle_grasp(side, buttons_map, g_pressed)

        if not transformations:
            return

        # Capture the data-read time once per tick; used as the message timestamp so
        # subscribers can compute accurate dt without relying on callback-execution time.
        stamp_ns = float(self.get_clock().now().nanoseconds)

        for side in self.sides:
            if side not in transformations:
                continue
            if side in self.gripper_only_sides:
                # Arm disabled: pin the delta to identity — same payload the
                # disengaged branch below publishes — so nothing downstream
                # can integrate hand motion into arm motion. Anchor stays
                # cleared, so the delta would start from identity if the arm
                # were ever re-enabled.
                identity = np.eye(4)
                self.anchors[side] = None
                self.last_delta[side] = None
                self.publish_transform(identity, self.child_frames[side])
                self.publish_matrix(identity, side, stamp_ns)
                continue
            raw = transformations[side]
            g_pressed = bool(buttons.get(self.trigger_keys[side], False))
            j_pressed = bool(buttons.get(self.joystick_keys[side], False))
            j_rising = j_pressed and not self.prev_j_pressed[side]
            self.prev_j_pressed[side] = j_pressed

            # Grasp handling happens at the top of timer_callback (see the
            # comment there); nothing to do per-side here.

            self._update_reference_frame(side, raw, g_pressed, j_pressed)

            # Apply VRPolicy-style calibration V, then ROS-axis alignment.
            pose = _M_ALIGN @ self.vr_to_global_mat[side] @ raw @ _M_ALIGN_INV

            # Move-enable Bool is already published unconditionally at the
            # top of timer_callback (see the comment there); no per-side
            # publish here to avoid double messages.

            if g_pressed:
                if self.anchors[side] is None:
                    # Rising edge of G: capture anchor pose.
                    self.anchors[side] = pose.copy()
                    self.last_delta[side] = np.eye(4)
                    self.get_logger().info(f'[{side}] trigger pressed — anchor captured')
                elif j_rising:
                    # J click during engagement: V was just re-captured, so
                    # re-capture the anchor too. The published delta resets
                    # to identity and tracks from this new reference.
                    self.anchors[side] = pose.copy()
                    self.last_delta[side] = np.eye(4)
                    self.get_logger().info(f'[{side}] reference reset via joystick click')
                # Delta expressed in the REFERENCE frame (set by joystick
                # click), not the anchor's local frame:
                #   translation = p_current - p_anchor        (plain difference)
                #   rotation    = R_current @ R_anchor^T      (left delta, about
                #                                              reference axes)
                # Starts at identity on grip press and describes the hand
                # motion w.r.t. the J-calibrated frame, independent of the
                # controller's orientation at grip time. (The former
                # inv(anchor) @ pose expressed everything in the grip-pose
                # axes, so gripping with a rotated wrist rotated the command
                # frame and bypassed the J reference.)
                anchor = self.anchors[side]
                delta = np.eye(4)
                delta[:3, :3] = pose[:3, :3] @ anchor[:3, :3].T
                delta[:3, 3] = pose[:3, 3] - anchor[:3, 3]
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
                if j_rising:
                    self.get_logger().info(f'[{side}] reference reset via joystick click')
                # While disengaged, publish identity so RViz still shows the frame.
                identity = np.eye(4)
                self.publish_transform(identity, self.child_frames[side])
                self.publish_matrix(identity, side, stamp_ns)

    def _handle_grasp(self, side, buttons_map, g_pressed):
        """Index trigger edge → grasp signal, GATED ON MOVE-ENABLE (G).

        CLOSE on press, OPEN on release, edge-triggered so the topic only
        sees state changes. Payload depends on --grasp-mode (see __init__).

        The gripper is only commandable while the arm is. With G released we
        swallow the trigger state into prev_grasp_pressed WITHOUT publishing,
        so:
        - the gripper holds whatever it was doing when G was released
          (grasping stays grasping — no accidental drop),
        - trigger presses/releases during the disengaged period are ignored,
        - re-engaging G does NOT resync the gripper to the trigger: prev
          already equals the current trigger, so no edge fires. The gripper
          next moves only on a real trigger transition made while engaged.
          (Resyncing instead would open the hand the instant you re-grip
          after letting the trigger go while clutching — exactly the drop
          this gate exists to prevent.)
        """
        grasp_pressed = bool(buttons_map.get(self.index_trigger_keys[side], False))
        if grasp_pressed == self.prev_grasp_pressed[side]:
            return
        self.prev_grasp_pressed[side] = grasp_pressed

        if not g_pressed:
            self.get_logger().info(
                f'[{side}] gripper trigger '
                f'{"pressed" if grasp_pressed else "released"} while '
                f'move-enable is OFF — ignored, gripper state held')
            return

        pub_pose = self.gripper_pose_publishers.get(side)
        if pub_pose is not None:
            pub_pose.publish(Int32(data=2 if grasp_pressed else 0))
        pub_do = self.do_grasp_publishers.get(side)
        if pub_do is not None:
            pub_do.publish(Bool(data=grasp_pressed))
        self.get_logger().info(
            f'[{side}] gripper {"CLOSE" if grasp_pressed else "OPEN"} '
            f'(mode={self.grasp_mode})')

    def _update_reference_frame(self, side, raw, g_pressed, j_pressed):
        """Update V = inv(raw); the world coordinate is SET ONLY by a joystick press.

        Follows DROID VRPolicy's "Update Definition Of Forward" block
        (archive/unused/ref/droid/droid/controllers/oculus_controller.py), except the
        grip trigger never locks the calibration:

        - While armed (before any J press), V tracks the current pose each
          tick, so every new grip engagement starts from identity.
        - Holding G freezes V for the duration of the movement (otherwise the
          delta would stay identity and nothing would move), but does NOT
          lock the calibration — tracking resumes on release.
        - Pressing J captures V = inv(raw) at that instant and LOCKS it:
          hold the controller level, pointing along the robot's forward
          axis, and click the joystick to define the world frame. Only the
          next J press changes it.
        """
        if j_pressed:
            self.reset_orientation[side] = True

        if not self.reset_orientation[side]:
            return

        if g_pressed and not j_pressed:
            # Moving without an explicit J: freeze V during the engagement,
            # but stay armed so tracking resumes when the grip is released.
            return

        try:
            V = np.linalg.inv(raw)
        except np.linalg.LinAlgError:
            # Singular raw pose; retry next tick.
            return

        self.vr_to_global_mat[side] = V
        if j_pressed:
            self.reset_orientation[side] = False

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
    import argparse
    parser = argparse.ArgumentParser(description='Oculus delta publisher (per-side).')
    parser.add_argument('--side', choices=('l', 'r', 'both'), default='both',
                        help='Which side(s) this node publishes for.')
    parser.add_argument('--grasp-mode',
                        choices=('do_grasp', 'pose_id', 'both'),
                        default='do_grasp',
                        help=('Trigger-edge output. do_grasp (default) -> Bool '
                              'on /dg_<side>/do_grasp with TRANSIENT_LOCAL QoS, '
                              'for no_teleop_grasp_control.py / online RL. '
                              'pose_id -> Int32 on /<side>/gripper_pose_id, for '
                              'glove_grasp_control.py. both -> publish both.'))
    parser.add_argument('--gripper-only', choices=('l', 'r', 'both'), default=None,
                        help=('Side(s) whose ARM is disabled: gripper still '
                              'follows the index trigger, but move_enable is '
                              'forced False and the delta stays identity, so '
                              'the servo never moves that arm. Both arms are '
                              'fully enabled by default.'))
    args, _ = parser.parse_known_args()
    sides = ('l', 'r') if args.side == 'both' else (args.side,)
    if args.gripper_only is None:
        gripper_only = ()
    elif args.gripper_only == 'both':
        gripper_only = ('l', 'r')
    else:
        gripper_only = (args.gripper_only,)

    rclpy.init()
    node = OculusDeltaControlNode(sides=sides, grasp_mode=args.grasp_mode,
                                  gripper_only_sides=gripper_only)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
