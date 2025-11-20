#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

using namespace std::chrono_literals;

class OffboardControl : public rclcpp::Node
{
public:
	OffboardControl() : Node("offboard_control_node")
	{
		RCLCPP_INFO(this->get_logger(), "Offboard Control Node Starting...");
		
		// Example Publisher (needed to prove dependencies work)
		offboard_control_mode_publisher_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
			"/fmu/in/offboard_control_mode", 10);
	}

private:
	rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
};

int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<OffboardControl>());
	rclcpp::shutdown();
	return 0;
}
