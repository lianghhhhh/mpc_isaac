from rclpy.node import Node
from nav_msgs.msg import Odometry

class CarStateNode(Node):
    def __init__(self):
        super().__init__('car_state_node')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.position = None
        self.orientation = None
        self.get_logger().info('CarStateNode has been started.')

    def odom_callback(self, msg):
        # Extract car state information from the Odometry message
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation

        # self.get_logger().info(f'Position: {self.position}, Orientation: {self.orientation}')