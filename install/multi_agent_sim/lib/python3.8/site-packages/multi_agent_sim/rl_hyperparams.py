# 1. import packages
import rclpy
from rclpy.node import Node

# 3. define Node class
class ParamServer(Node):
    def __init__(self):
        # allow_undeclared_parameters is True for allowing to delete parameters
        super().__init__("rl_training_params_server", allow_undeclared_parameters=True)
        self.get_logger().info("SAC hyper-parameters server")

    def declare_param(self):
        self.get_logger().info("----new declared param----")
        self.declare_parameter("car_name", "tiger")
        self.declare_parameter("width", 1.55)
        self.declare_parameter("wheels", 5)

    def get_param(self):
        self.get_logger().info("----query param----")
        # get specific parameter
        car_name = self.get_parameter("car_name")
        self.get_logger().info("%s = %s" %(car_name.name, car_name.value))
        # get multiple parameters
        params = self.get_parameters(["car_name", "width", "wheels"])
        for param in params:
            self.get_logger().info("%s = %s" % (param.name, param.value))
        # check if the parameter is contained in server
        self.get_logger().info("Does it contain car_name? %d" % self.has_parameter("car_name"))
        self.get_logger().info("Does it contain height? %d" % self.has_parameter("height"))

    def update_param(self):
        self.get_logger().info("----update param----")
        self.set_parameters([rclpy.Parameter("car_name", value="f1tenth")])
        car_name = self.get_parameter("car_name")
        self.get_logger().info("Modified car_name is %s = %s" % (car_name.name, car_name.value))

    def delete_param(self):
        self.get_logger().info("----delete param----")
        self.get_logger().info("Does it contain car_name? %d" % self.has_parameter("car_name"))
        self.undeclare_parameter("car_name")
        self.get_logger().info("Does it contain car_name? %d" % self.has_parameter("car_name"))

def main():
    # 2. ROS node initialization
    rclpy.init()

    # 4. spin() - input Node class object
    node = ParamServer()
    node.declare_param()
    node.get_param()
    node.update_param()
    node.delete_param()
    rclpy.spin(node)

    # 5. release resource
    rclpy.shutdown()


if __name__ == '__main__':
    main()