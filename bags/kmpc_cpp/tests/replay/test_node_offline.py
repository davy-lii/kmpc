#!/usr/bin/env python3
"""End-to-end ROS test; run ONLY in the documented --network none container.
The fake feedback follows debug commands. No robot driver/device is started.
"""
import os, sys, json, time, math, subprocess
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose, PoseArray

root=Path(sys.argv[1]);fixture=json.loads((root/'node_fixture.json').read_text())
os.environ['ROS_DOMAIN_ID']='213'
build=Path(os.environ.get('KMPC_BUILD_DIR',str(root/'build')))
node_process=subprocess.Popen([str(build/'kmpc_node'),'--ros-args','-p','auto_enable:=false','-p','require_xoip:=false','-p','line_dq_scale:=0.3','-p','stream_debug_q_hz:=250.0'],stdout=(root/'node_runtime.log').open('w'),stderr=subprocess.STDOUT)
rclpy.init();node=Node('offline_robot_feedback')
q=list(map(float,fixture['q']));cmd=list(q);bias=0.0;statuses=[];commands=[]
pubq=node.create_publisher(JointState,'/joint_states',10)
puben=node.create_publisher(Bool,'/kmpc/enable',10)
pubseg=node.create_publisher(PoseArray,'/kmpc/segment_left',10)
names=['Chassis_Y_Joint1','Chassis_Y_Joint2','Chassis_Y_Joint3','Chassis_Z_Joint4']+[f'{side}Arm_Z_Joint{i}' for side in ['Right','Left'] for i in range(1,8)]
def feedback():
 global q
 q=list(cmd);q[12]+=bias
 m=JointState();m.name=names;m.position=q;m.header.stamp=node.get_clock().now().to_msg();pubq.publish(m)
def oncmd(m):
 global cmd
 if len(m.position)==18:
  cmd=list(m.position);commands.append((time.monotonic(),list(cmd)))
def onstatus(m):
 tokens={}
 for part in m.data.split():
  if '=' in part:
   k,v=part.split('=',1);tokens[k]=v
 statuses.append((time.monotonic(),tokens))
node.create_subscription(JointState,'/kmpc/debug/q',oncmd,100)
node.create_subscription(String,'/kmpc/status',onstatus,100)
node.create_timer(.004,feedback)
def spin(seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  if node_process.poll() is not None:raise RuntimeError('kmpc_node exited')
  rclpy.spin_once(node,timeout_sec=.004)
def wait(pred,seconds=15):
 start=len(statuses);end=time.monotonic()+seconds
 while time.monotonic()<end:
  spin(.02)
  for _,s in statuses[start:]:
   if pred(s):return s
  start=len(statuses)
 raise AssertionError('status wait timed out: '+str(statuses[-1:]));
# Input is tool orientation; multiply flange quaternion by inverse RotZ(90deg).
x,y,z,w=fixture['flange_quat'];a=math.sqrt(.5)
tool=[(x-y)*a,(x+y)*a,(z-w)*a,(w+z)*a]
base=list(map(float,fixture['position']));goal=[base[0],base[1]+.06,base[2]]
def segment(generation,xyz=None,tip=None):
 m=PoseArray();m.header.frame_id='base_link';m.header.stamp.sec=generation//1000000000;m.header.stamp.nanosec=generation%1000000000
 for p in ([xyz]+([tip] if tip else [])) if xyz is not None else []:
  v=Pose();v.position.x,v.position.y,v.position.z=p;v.orientation.x,v.orientation.y,v.orientation.z,v.orientation.w=tool;m.poses.append(v)
 pubseg.publish(m)
results=[]
try:
 spin(1);en=Bool();en.data=True;puben.publish(en);spin(.3)
 segment(10,goal,[base[0],base[1]+.075,base[2]])
 wait(lambda s:s.get('goal_generation')=='10' and s.get('line')=='on')
 bias=.02
 wait(lambda s:s.get('goal_generation')=='10' and s.get('line')=='hold',20)
 spin(.25)
 assert statuses[-1][1].get('arrived')=='False',statuses[-1]
 results.append('reference endpoint alone does not report measured arrival')
 bias=0
 wait(lambda s:s.get('goal_generation')=='10' and s.get('arrived')=='True')
 results.append('matching measured endpoint reports arrival')
 segment(10,[goal[0]+.01,goal[1],goal[2]])
 wait(lambda s:s.get('goal_generation')=='10' and s.get('line')=='hold' and s.get('frozen')=='1')
 segment(10,[goal[0]+.05,goal[1],goal[2]])
 wait(lambda s:s.get('goal_generation')=='10' and s.get('line')=='stop' and s.get('abort')=='1')
 stopped=list(cmd);segment(10,goal);spin(.25)
 assert statuses[-1][1].get('line')=='stop'
 assert max(abs(a-b) for a,b in zip(stopped,cmd))<1e-8
 results.append('same-generation 50 mm jump latches stop; drift cannot rearm')
 segment(11,goal)
 wait(lambda s:s.get('goal_generation')=='11' and s.get('line')=='hold' and s.get('arrived')=='True')
 segment(10);spin(.2)
 assert statuses[-1][1].get('goal_generation')=='11' and statuses[-1][1].get('line')=='hold'
 results.append('delayed clear for old generation cannot erase next segment')
 segment(11)
 wait(lambda s:s.get('goal_generation')=='11' and s.get('line')=='off')
 results.append('matching clear ends the segment')
 # Replace a pending plan immediately; stale worker results must not execute.
 segment(12,[goal[0]+.15,goal[1],goal[2]])
 spin(.02);segment(12);segment(13,goal)
 wait(lambda s:s.get('goal_generation')=='13' and s.get('line')=='hold' and s.get('arrived')=='True',20)
 results.append('pending plan cancellation and replacement stays bounded')
 active=sorted(float(s.get('solve_ms','0')) for _,s in statuses if s.get('line')=='on' and float(s.get('solve_ms','0'))>0)
 def percentile(values,p):return values[min(len(values)-1,int(p*(len(values)-1)))] if values else 0
 report={'passed':results,'statuses':len(statuses),'stream_samples':len(commands),'max_tick_solve_ms':max(float(s.get('solve_ms','0')) for _,s in statuses),'line_tick_p95_ms':percentile(active,.95),'line_tick_p99_ms':percentile(active,.99)}
 (root/'node_offline_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:
 node_process.terminate()
 try:node_process.wait(timeout=5)
 except subprocess.TimeoutExpired:node_process.kill();node_process.wait()
 node.destroy_node();rclpy.shutdown()
