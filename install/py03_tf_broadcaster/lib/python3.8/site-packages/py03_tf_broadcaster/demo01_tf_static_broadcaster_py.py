
import rclpy
from rclpy.node import Node
import sys
from rclpy.logging import get_logger
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations

class TFStaticBroadcasterPy(Node):
    def __init__(self):
        super().__init__("tf_static_broadcaster_py_node")

        # create tf broadcaster
        self.broadcaster =  StaticTransformBroadcaster(self)
        # publish tf
        list = [0.0, 0.0, 0.0, 0.5, 0.5, 0.2, 0.4]
        self.pub_static_tf(list)

    def pub_static_tf(self, list):
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = 'base_link'
        ts.child_frame_id = 'laser'

        ts.transform.translation.x = float(list[1])
        ts.transform.translation.y = float(list[2])
        ts.transform.translation.z = float(list[3])

        qtn = tf_transformations.quaternion_from_euler(
            float(list[4]),
            float(list[5]),
            float(list[6])
        )
        # qtn = [1.0, 2.0, 3.0, 4.0]
        ts.transform.rotation.x = qtn[0]
        ts.transform.rotation.y = qtn[1]
        ts.transform.rotation.z = qtn[2]
        ts.transform.rotation.w = qtn[3]

        self.broadcaster.sendTransform(ts)

def main():

    # if len(sys.argv) != 9:
    #     get_logger("rclpy").info("The number of arguments passes is illegal!")
    #     return
    print("run here")
    rclpy.init()
    rclpy.spin(TFStaticBroadcasterPy())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
