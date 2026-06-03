from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class PathPointsNode(Node):
    def __init__(self):
        super().__init__('path_points_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/nearest_curve_point',
            self.path_points_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.nearest_points = None  # Will hold the nearest path points
        self.get_logger().info('PathPointsNode has been started.')

    def path_points_callback(self, msg):
        # Store the received path point
        data = msg.data # [idx, x, y, angle]
        if len(data) >= 10:
            for i in range(0, 3):
                setattr(self, f'point_{i}', NearestPoint(x=data[1 + i*3], y=data[2 + i*3], angle=data[3 + i*3]))
            self.nearest_points = [getattr(self, f'point_{i}') for i in range(3)]
        else:
            self.nearest_points = None
        # self.get_logger().info(f'Received Nearest Path Point: {self.nearest_point}')

class NearestPoint:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.x = x
        self.y = y
        self.angle = angle