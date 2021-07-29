#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
#from generate_pattern import aruco



def talker():
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    rospy.init_node('goal_node')

    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.pose.orientation.w = 1.0
    goal_pose.pose.orientation.x = 0.0
    goal_pose.pose.orientation.y = 0.0
    goal_pose.pose.orientation.z = 0.0

    goal_pose.pose.position.x = 0.0
    goal_pose.pose.position.y = 0.0
    goal_pose.pose.position.z = 0.0

    #pub.publish(goal_pose)

    rate = rospy.Rate(1) # 10hz
    
    while not rospy.is_shutdown():
        rate.sleep()
        pub.publish(goal_pose)

if __name__ == '__main__':
    try:
        markerLen = 70/1000
        talker()

    except rospy.ROSInterruptException:
        pass
