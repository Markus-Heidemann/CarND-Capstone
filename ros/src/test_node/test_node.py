#!/usr/bin/env python

import math
import rospy

def test_node():
	rospy.logfatal('======== TEST_NODE WAS CALLED ========')

class Test_Node(object):
    def __init__(self):
        rospy.init_node('test_node')
        rospy.logfatal('======== TEST_NODE WAS CALLED ========')
        rospy.spin()

if __name__ == '__main__':
    try:
        Test_Node()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start test_node.')
