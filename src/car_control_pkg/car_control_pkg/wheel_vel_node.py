from rclpy.node import Node
from sensor_msgs.msg import JointState

class WheelVelNode(Node):
    def __init__(self):
        super().__init__('wheel_vel_node')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.wheel_velocities = None
        self.get_logger().info('WheelVelNode has been started.')

    def joint_state_callback(self, msg):
        # Extract wheel velocities from the JointState message
        self.wheel_velocities = {}
        for name, velocity in zip(msg.name, msg.velocity):
            if 'Revolute' in name:
                self.wheel_velocities[name] = velocity
        # self.get_logger().info(f'Wheel Velocities: {self.wheel_velocities}')