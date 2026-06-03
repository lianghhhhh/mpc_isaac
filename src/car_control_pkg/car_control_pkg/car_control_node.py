import joblib
import numpy as np
import time
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from car_control_pkg.utils import loadModelFunc, createMpcSolver, normalize, denormalize, createAcadosSolver

class CarControlNode(Node):
    def __init__(self, car_state_node, path_points_node, wheel_vel_node, model_path, N_steps, dt):
        super().__init__('car_control_node')
        self.get_logger().info('Car Control Node has been started.')
        self.front_wheel_pub = self.create_publisher(Float64MultiArray, "car_C_front_wheel", 10)
        self.rear_wheel_pub = self.create_publisher(Float64MultiArray, "car_C_rear_wheel", 10)
        # self.wheel_vel_pub = self.create_publisher(Float32MultiArray, "wheel_velocity", 10)
        self.pred_path_pub = self.create_publisher(Float64MultiArray, "predicted_path", 10)

        self.car_state_node = car_state_node
        self.path_points_node = path_points_node
        self.wheel_vel_node = wheel_vel_node

        self.model_path = model_path
        # self.x_scaler = joblib.load(f'{self.model_path}/x_scaler.save')
        # self.u_scaler = joblib.load(f'{self.model_path}/u_scaler.save')

        self.N_steps = N_steps
        self.dt = dt
        self.prev_u = np.zeros((4,))
        self.delay = 0.02  # seconds
        self._solving = False
        self._timer_period = 0.05  # 20 Hz
        self._cycle_count = 0

        # MPC/cache handles
        self._model_func, self._lib_dir, self._lib_name = loadModelFunc(self.model_path, self.dt)
        # self._solver, self._u_pred, self._next_x_pred, self._current_x, self._target_path = createMpcSolver(self._model_func, N=10)
        self._acados_solver = createAcadosSolver(self._model_func, self._lib_dir, self._lib_name, self.N_steps, self.dt)
        
        # Run MPC periodically while the node is spinning (20 Hz)
        self.create_timer(self._timer_period, self.find_control_command)

    def _build_reference_path(self, nearest_points, horizon_size):
        path = np.array([[point.x, point.y] for point in nearest_points], dtype=float)

        if path.shape[0] >= horizon_size:
            return path[:horizon_size]

        if len(nearest_points) >= 2:
            step = np.array(
                [nearest_points[-1].x - nearest_points[-2].x, nearest_points[-1].y - nearest_points[-2].y],
                dtype=float,
            )
            if np.linalg.norm(step) < 1e-6:
                step = np.array(
                    [np.cos(nearest_points[-1].angle), np.sin(nearest_points[-1].angle)],
                    dtype=float,
                ) * 0.1
        else:
            step = np.array(
                [np.cos(nearest_points[-1].angle), np.sin(nearest_points[-1].angle)],
                dtype=float,
            ) * 0.1

        while path.shape[0] < horizon_size:
            path = np.vstack([path, path[-1] + step])

        return path

    def publish_control_command(self, control_input):
        self.get_logger().info(f'Publishing control command: {control_input}')
        front_msg = Float64MultiArray()
        rear_msg = Float64MultiArray()
        vals = [float(-x) for x in control_input] # flip control direction if needed
        # msg = Float64MultiArray()
        # vals = [float(-x) for x in control_input] # flip control direction if needed
        # msg.data = vals
        front_msg.data = vals[0:2]
        rear_msg.data = vals[2:4]
        self.front_wheel_pub.publish(front_msg)
        self.rear_wheel_pub.publish(rear_msg)
        # self.wheel_vel_pub.publish(msg)

    def publish_predicted_path(self, predicted_path):
        msg = Float64MultiArray()
        vals = predicted_path.flatten().tolist()
        msg.data = vals
        self.pred_path_pub.publish(msg)

    def find_control_command(self):
        if self._solving:
            return

        self._solving = True
        cycle_start = time.perf_counter()

        pos = self.car_state_node.position
        ori = self.car_state_node.orientation
        if pos is None or ori is None:
            self.get_logger().warn('Invalid car state received: position/orientation missing')
            self._solving = False
            return

        wheel_velocities = self.wheel_vel_node.wheel_velocities
        if wheel_velocities is None:
            self.get_logger().warn('Wheel velocities not received yet.')
            self._solving = False
            return
        
        vel_left = wheel_velocities.get('Revolute_3', 0.0)
        vel_right = wheel_velocities.get('Revolute_4', 0.0)
        # Quaternion to yaw (assumes geometry_msgs/Quaternion: x,y,z,w)
        yaw = self.compute_angle(ori.x, ori.y, ori.z, ori.w)
        current_state = np.array([pos.x, pos.y, np.sin(yaw), np.cos(yaw), vel_left, vel_right], dtype=float)

        nearest_points = self.path_points_node.nearest_points
        if nearest_points is None or len(nearest_points) == 0:
            self.get_logger().warn('No nearest path points received yet.')
            self._solving = False
            return

        path_points = self._build_reference_path(nearest_points, self.N_steps + 1)
        if len(path_points) != (self.N_steps + 1):
            self.get_logger().warn(f'Invalid path points received: {path_points}')
            self._solving = False
            return
        
        # use current state and path points to compute optimal control
        # current_state = normalize(current_state, "state", self.x_scaler)
        # path_points = normalize(path_points, "path", self.x_scaler)
        # state_data = ca.DM(current_state)
        # target_path_data = ca.DM(path_points)

        # self._solver.set_value(self._current_x, state_data)
        # self._solver.set_value(self._target_path, target_path_data)
        # Set initial state constraints (only for stage 0)
        self._acados_solver.set(0, "lbx", current_state)
        self._acados_solver.set(0, "ubx", current_state)
        
        # OPTIMIZATION: Only initialize the full horizon on the very first frame.
        # Subsequent steps automatically reuse the solver's internal trajectory.
        if self._cycle_count == 0:
            for t in range(self.N_steps + 1):
                self._acados_solver.set(t, "x", current_state)
            for t in range(self.N_steps):
                self._acados_solver.set(t, "u", self.prev_u)
        else:
            # Quick warm start for stage 0 control input
            self._acados_solver.set(0, "u", self.prev_u)

        # Set target path parameters across the horizon
        for t in range(self.N_steps + 1):
            self._acados_solver.set(t, "p", path_points[t, :])

        try:
            # solution = self._solver.solve()
            # optimal_control = solution.value(self._u_pred)
            # control_input = optimal_control[0, :]
            # control_input = denormalize(control_input, "u", self.u_scaler)
            # self.publish_control_command(control_input)
            status = self._acados_solver.solve()
            if status != 0:
                self.get_logger().error(f'Acados solver failed with status {status}')
                self._solving = False
                return

            control_input = self._acados_solver.get(0, "u") # get optimal control at time 0
            self.prev_u = control_input  # store for warm starty
            # control_input = denormalize(control_input, "u", self.u_scaler)
            predict_path = np.array([self._acados_solver.get(t, "x") for t in range(self.N_steps + 1)])
            # predict_path = denormalize(predicted_states[:, :2], "x", self.x_scaler)
            
            self.publish_control_command(control_input)
            self.publish_predicted_path(predict_path)
            self._cycle_count += 1

            elapsed_ms = (time.perf_counter() - cycle_start) * 1000.0
            if elapsed_ms > 50.0:
                self.get_logger().warn(f'MPC cycle over budget: {elapsed_ms:.1f} ms')

        except Exception as e:
            self.get_logger().error(f'MPC solver failed: {e}')
        finally:
            self._solving = False

    def compute_angle(self, qx, qy, qz, qw):
        # Convert quaternion to yaw angle in radians
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        radian = np.arctan2(siny_cosp, cosy_cosp)
        
        # Flip orientation by 180 degrees
        radian = radian + np.pi
        if radian > np.pi:
            radian -= 2 * np.pi
        
        return radian