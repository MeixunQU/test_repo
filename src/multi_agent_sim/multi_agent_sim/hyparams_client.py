# 1. import packages
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import ListParameters
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterValue
from ros2param.api import get_parameter_value

class MinimalParamClient(Node):

    def __init__(self):
        super().__init__('minimal_param_client_py')

    def list_params(self):
        # 3-1.create cient
        cli_list = self.create_client(ListParameters, '/param_server_node_py/list_parameters')
        # 3-2.wait for the service connection
        while not cli_list.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Listing the parameters, please wait...')
        req = ListParameters.Request()
        future = cli_list.call_async(req)
        rclpy.spin_until_future_complete(self,future)
        return future.result()

    def get_params(self,names):
        # 3-1.create client
        cli_get = self.create_client(GetParameters, '/param_server_node_py/get_parameters')
        # 3-2.wait for the service connection
        while not cli_get.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Listing the parameters, please wait...')
        req = GetParameters.Request()
        req.names = names
        future = cli_get.call_async(req)
        rclpy.spin_until_future_complete(self,future)
        return future.result()

    def set_params(self):
        # 3-1.create client
        cli_set = self.create_client(SetParameters, '/param_server_node_py/set_parameters')
        # 3-2. wait for the service connection
        while not cli_set.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Listing the parameters, please wait...')

        req = SetParameters.Request()

        p1 = Parameter()
        p1.name = "car_type"

        # v1 = ParameterValue()
        # v1.type = ParameterType.PARAMETER_STRING
        # v1.string_value = "Pig"
        # p1.value = v1
        p1.value = get_parameter_value(string_value="Pig")

        p2 = Parameter()
        p2.name = "height"

        v2 = ParameterValue()
        v2.type = ParameterType.PARAMETER_DOUBLE
        v2.double_value = 0.3
        p2.value = v2
        # p2.value = get_parameter_value(string_value="0.3")

        req.parameters = [p1, p2]
        future = cli_set.call_async(req)
        rclpy.spin_until_future_complete(self,future)
        return future.result()

def main():
    # 2. ROS2 Node initialisation
    rclpy.init()
    # 4. create object
    client = MinimalParamClient()

    # get params list
    client.get_logger().info("---------get params list---------")
    response = client.list_params()
    for name in response.result.names:
        client.get_logger().info(name)

    client.get_logger().info("---------get params---------")
    names = ["height","car_type", "width"]
    response = client.get_params(names)
    # print(response.values)
    for v in response.values:
        if v.type == ParameterType.PARAMETER_STRING:
            client.get_logger().info("string:%s" % v.string_value)
        elif v.type == ParameterType.PARAMETER_DOUBLE:
            client.get_logger().info("float:%.2f" % v.double_value)

    client.get_logger().info("---------set params---------")
    response = client.set_params()
    results = response.results
    client.get_logger().info("Set %d params" % len(results))
    for result in results:
        if not result.successful:
            client.get_logger().info("Failed params setting")
    rclpy.shutdown()

if __name__ == "__main__":
    main()