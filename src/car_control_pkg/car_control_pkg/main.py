import rclpy
from car_control_pkg.utils import loadConfig
from rclpy.executors import MultiThreadedExecutor
from car_control_pkg.car_state_node import CarStateNode
from car_control_pkg.car_control_node import CarControlNode
from car_control_pkg.path_points_node import PathPointsNode
from car_control_pkg.wheel_vel_node import WheelVelNode

def main():
    rclpy.init()

    config = loadConfig()

    car_state_node = CarStateNode()
    path_points_node = PathPointsNode()
    wheel_vel_node = WheelVelNode()
    car_control_node = CarControlNode(car_state_node, path_points_node, wheel_vel_node,
                                      config['model_path'], config['N_steps'], config['dt'])

    executor = MultiThreadedExecutor()
    executor.add_node(car_state_node)
    executor.add_node(path_points_node)
    executor.add_node(wheel_vel_node)
    executor.add_node(car_control_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
