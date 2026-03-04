#!/usr/bin/env python3
"""
CARLA Realistic Bridge - Digital Twin Emulator
Description: Bridges CARLA simulation with real vehicle control logic by 
emulating physical hardware constraints (PID motor control and valve hysteresis).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64, Bool
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
import math
import numpy as np
import time

# --- REAL VEHICLE PHYSICAL PARAMETERS ---
TRANSMISSION_RATIO = 14
STEP_ANGLE = 1.8 
MAX_STEER_ANGLE = 10.0  # Physical mechanical limit in degrees

class PIDController:
    """Standard PID Controller mirroring the real vehicle's steering logic."""
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, target, actual, dt):
        error = target - actual
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class CarlaRealisticBridge(Node):
    def __init__(self):
        super().__init__('carla_realistic_bridge')
        
        # 1. STEERING EMULATION (STEPPER MOTOR)
        # Tune these PID gains to match your physical motor's response time
        self.pid = PIDController(kp=1.0, ki=0.0, kd=0.0) 
        self.target_angle = 0.0
        self.current_motor_pos = 0.0  # Simulated motor position in microsteps
        
        # 2. INTERNAL STATE VARIABLES
        self.current_speed_kmh = 0.0
        self.stop = False
        self.throttle_cmd = 0.0  # Input from the external Throttle Node (0 or 50)
        
        # --- SUBSCRIPTIONS ---
        self.create_subscription(Float64, 'commands/KalmanAngle', self.steering_cb, 10)
        self.create_subscription(Float32, '/ECU/throttle', self.throttle_cb, 10)
        self.create_subscription(Bool, 'commands/stop', self.stop_cb, 10)
        self.create_subscription(Odometry, '/carla/hero/odometry', self.odom_cb, 10)

        # --- PUBLISHERS ---
        self.control_pub = self.create_publisher(CarlaEgoVehicleControl, '/carla/hero/vehicle_control_cmd', 10)
        self.speed_pub = self.create_publisher(Float32, '/ECU/speed', 10)

        # Control Loop at 100Hz (Matches the real hardware's PID frequency)
        self.create_timer(0.01, self.physics_loop)
        self.get_logger().info("Realistic CARLA Bridge initialized with Hardware Emulation.")

    def angle_to_pos(self, angle):
        """Converts steering angle (degrees) to motor microsteps."""
        steps = (angle / STEP_ANGLE) * 256
        return -int(steps * TRANSMISSION_RATIO)

    def steering_cb(self, msg):
        """Update target angle with safety clipping."""
        self.target_angle = np.clip(msg.data, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)

    def throttle_cb(self, msg):
        """Capture the throttle command (expects 0.0 or 50.0)."""
        self.throttle_cmd = msg.data

    def stop_cb(self, msg):
        self.stop = msg.data

    def odom_cb(self, msg):
        """Process speed feedback and convert to km/h for the control logic."""
        vel = msg.twist.twist.linear
        speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        self.current_speed_kmh = speed_ms * 3.6
        
        # Publish converted speed for the Throttle/ECU node
        self.speed_pub.publish(Float32(data=self.current_speed_kmh))

    def physics_loop(self):
        """Main physics loop simulating real-world hardware latency."""
        dt = 0.01 
        
        # --- STEERING MOTOR EMULATION ---
        target_pos = self.angle_to_pos(self.target_angle)
        pid_output = self.pid.compute(target_pos, self.current_motor_pos, dt)
        
        # Increment simulated motor position (hardware doesn't teleport!)
        self.current_motor_pos += pid_output
        
        # Normalize motor position to CARLA's -1.0 to 1.0 steering range
        max_limit = abs(self.angle_to_pos(MAX_STEER_ANGLE))
        carla_steer = np.clip(self.current_motor_pos / max_limit, -1.0, 1.0)

        # --- THROTTLE & BRAKE EMULATION ---
        # Map 0-100 throttle range to CARLA's 0.0-1.0 range
        carla_throttle = (self.throttle_cmd / 100.0) if not self.stop else 0.0
        carla_brake = 1.0 if self.stop else 0.0

        # --- CONSTRUCT AND SEND COMMAND ---
        msg = CarlaEgoVehicleControl()
        msg.steer = float(carla_steer)
        msg.throttle = float(carla_throttle)
        msg.brake = float(carla_brake)
        msg.hand_brake = False
        msg.manual_gear_shift = False
        self.control_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CarlaRealisticBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Bridge shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()