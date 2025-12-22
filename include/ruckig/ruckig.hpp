#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <iostream>
#include <limits>
#include <math.h>
#include <numeric>
#include <optional>
#include <tuple>

#include <ruckig/calculator.hpp>
#include <ruckig/error.hpp>
#include <ruckig/input_parameter.hpp>
#include <ruckig/output_parameter.hpp>
#include <ruckig/trajectory.hpp>


namespace ruckig {

//! Main interface for the Ruckig algorithm
/*
    template<class, size_t> class这是一个整体，它描述了 CustomVector这个参数所接受的“模板”必须长什么样子。
    它声明了 CustomVector必须是一个类模板，并且这个类模板需要接受两个模板参数：
    1.一个类型参数（用 class或 typename声明）。
    2.一个非类型参数，具体是 size_t类型的。
    CustomVector是这个模板参数的名称。在 Ruckig类的内部，可以像使用一个普通的类模板一样使用 CustomVector
    我（Ruckig类）需要一个容器来存放数据，但我不想固定死用哪种容器（比如 std::vector或 std::array）。
    我只规定这个容器必须能通过 Container<ElementType, Size>这样的形式来创建。具体你用哪个容器，由使用我的人来决定。
*/

template<size_t DOFs = 0, template<class, size_t> class CustomVector = StandardVector, bool throw_error = false>
class Ruckig {
    //! Current input, only for comparison for recalculation
    InputParameter<DOFs, CustomVector> current_input;

    //! Flag that indicates if the current_input was properly initialized
    bool current_input_initialized {false};

public:
    //! Calculator for new trajectories
    Calculator<DOFs, CustomVector> calculator;

    //! Max number of intermediate waypoints
    const size_t max_number_of_waypoints;

    //! Degrees of freedom
    const size_t degrees_of_freedom;

    //! Time step between updates (cycle time) in [s]
    double delta_time {0.0};


    /*
    SFINAE（Substitution Failure Is Not An Error）是C++模板元编程的一项重要原则。简单来说，就是在模板参数推导和替换过程中，
    如果某个替换失败了，编译器不会直接报错，而是安静地将这个模板特例从候选重载集合中移除，然后继续尝试其他可能匹配的特例。

    SFINAE机制：typename std::enable_if<(D >= 1), int>::type = 0是开关。当 D >= 1为真时，
    std::enable_if会有一个 int类型的 type成员，这个构造函数是有效的。如果条件不满足，这个构造函数会从重载集中被剔除。
    */
    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit Ruckig():
        max_number_of_waypoints(0),
        degrees_of_freedom(DOFs),
        delta_time(-1.0)
    {
    }


    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit Ruckig(double delta_time):
        max_number_of_waypoints(0),
        degrees_of_freedom(DOFs),
        delta_time(delta_time)
    {
    }

#if defined WITH_CLOUD_CLIENT
    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit Ruckig(double delta_time, size_t max_number_of_waypoints):
        current_input(InputParameter<DOFs, CustomVector>(max_number_of_waypoints)),
        calculator(Calculator<DOFs, CustomVector>(max_number_of_waypoints)),
        max_number_of_waypoints(max_number_of_waypoints),
        degrees_of_freedom(DOFs),
        delta_time(delta_time)
    {
    }
#endif

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit Ruckig(size_t dofs):
        current_input(InputParameter<DOFs, CustomVector>(dofs)),
        calculator(Calculator<DOFs, CustomVector>(dofs)),
        max_number_of_waypoints(0),
        degrees_of_freedom(dofs),
        delta_time(-1.0)
    {
    }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit Ruckig(size_t dofs, double delta_time):
        current_input(InputParameter<DOFs, CustomVector>(dofs)),
        calculator(Calculator<DOFs, CustomVector>(dofs)),
        max_number_of_waypoints(0),
        degrees_of_freedom(dofs),
        delta_time(delta_time)
    {
    }

#if defined WITH_CLOUD_CLIENT
    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit Ruckig(size_t dofs, double delta_time, size_t max_number_of_waypoints):
        current_input(InputParameter<DOFs, CustomVector>(dofs, max_number_of_waypoints)),
        calculator(Calculator<DOFs, CustomVector>(dofs, max_number_of_waypoints)),
        max_number_of_waypoints(max_number_of_waypoints),
        degrees_of_freedom(dofs),
        delta_time(delta_time)
    {
    }
#endif

    //! Reset the instance (e.g. to force a new calculation in the next update)
    void reset() {
        current_input_initialized = false;
    }

    //! Filter intermediate positions based on a threshold distance for each DoF
    /*
    1.函数作用：删除“几何上冗余”的中间路径点（waypoints），但保证轨迹误差不超过给定阈值。如果中间点太密、太接近直线会 人为制造多次加减速，降低时间最优性，放大数值噪声以及实时性不友好。所以 在真正做 OTG 之前，必须先“瘦身”路径点；剪枝优化；
    2.函数总体流程：
        1. 如果没有中间点 → 直接返回
        2. 初始化所有路径点为“保留”
        3. 用滑动窗口 (start, end) 扫描路径
        4. 检查：start → end 是否能“覆盖”中间所有点
        5. 如果能覆盖 → 中间点标记为冗余
        6. 如果不能 → 保留该点，窗口前移
        7. 收集所有被保留的点并返回
    3.

    */
    template<class T> using Vector = CustomVector<T, DOFs>;
    std::vector<Vector<double>> filter_intermediate_positions(const InputParameter<DOFs, CustomVector>& input, const Vector<double>& threshold_distance) const {
        // 1.如果没中间点，直接返回，避免后面大量计算
        if (input.intermediate_positions.empty()) {
            return input.intermediate_positions;
        }

        // 2.初始化所有路径点为“保留”
        const size_t n_waypoints = input.intermediate_positions.size(); //intermediate_positions如何在InputParameter中初始化的？似乎没有给出具体大小？
        std::vector<bool> is_active;
        is_active.resize(n_waypoints);
        for (size_t i = 0; i < n_waypoints; ++i) {
            is_active[i] = true;
        }

        // 3.用滑动窗口 (start, end) 扫描路径，[start] —— [中间点] —— [end]
        size_t start = 0;
        size_t end = start + 2;
        for (; end < n_waypoints + 2; ++end) {
            // 路径边界情况 三目运算符 const auto 变量 = (条件) ? 条件为真时取的值 : 条件为假时取的值；
            const auto pos_start = (start == 0) ? input.current_position : input.intermediate_positions[start - 1];
            const auto pos_end = (end == n_waypoints + 1) ? input.target_position : input.intermediate_positions[end - 1];

            // Check for all intermediate positions
            bool are_all_below {true};
            for (size_t current = start + 1; current < end; ++current) {
                const auto pos_current = input.intermediate_positions[current - 1];

                // Is there a point t on the line that holds the threshold?
                double t_start_max = 0.0;
                double t_end_min = 1.0;
                for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                    /*当h0处于[0,1]区间外时，说明该点在start<-->end外部区域，严格意义上该点是不可以被覆盖的，但是这样子会导致过多的路径点被保留，影响后续OTG性能，因此引入threshold_distance，如果点在外部区域但是距离线段足够近的话，也允许被覆盖*/
                    
                    // 公式推导：pos_current = pos_start + h0 * (pos_end - pos_start)  一维点举例，可拓展到多维
                    const double h0 = (pos_current[dof] - pos_start[dof]) / (pos_end[dof] - pos_start[dof]); 
                    // 公式推导
                    // 工程上允许：|p(t) - pos_current | <= threshold_distance 又p(t) = pos_start + t * (pos_end - pos_start)
                    // 即：| pos_start + t * (pos_end - pos_start) - pos_current | <= threshold_distance，又pos_current = pos_start + h0 * (pos_end - pos_start)
                    // 化简得：| t - h0 | * | pos_end - pos_start | <= threshold_distance
                    // 即：| t - h0 | <= threshold_distance / | pos_end - pos_start |
                    // 即：t ∈ [ h0 - threshold_distance / | pos_end - pos_start | ,  h0 + threshold_distance / | pos_end - pos_end - pos_start | ] 
                    const double t_start = h0 - threshold_distance[dof] / std::abs(pos_end[dof] - pos_start[dof]);  //注意分母为0？？？
                    const double t_end = h0 + threshold_distance[dof] / std::abs(pos_end[dof] - pos_start[dof]);

                    t_start_max = std::max(t_start, t_start_max);   //区间左边界，需满足[0,1]区间内
                    t_end_min = std::min(t_end, t_end_min);         //区间右边界，需满足[0,1]区间内

                    if (t_start_max > t_end_min) {
                        are_all_below = false;
                        break;
                    }
                }
                if (!are_all_below) {
                    break;
                }
            }

            is_active[end - 2] = !are_all_below;
            if (!are_all_below) {
                start = end - 1;
            }
        }

        std::vector<Vector<double>> filtered_positions;
        filtered_positions.reserve(n_waypoints);
        for (size_t i = 0; i < n_waypoints; ++i) {
            if (is_active[i]) {
                filtered_positions.push_back(input.intermediate_positions[i]);
            }
        }

        return filtered_positions;
    }

    //! Validate the input as well as the Ruckig instance for trajectory calculation
    //检查输入参数的合法性
    template<bool throw_validation_error = true>    //模板默认参数,编译期策略开关，并非运行期开关
    bool validate_input(const InputParameter<DOFs, CustomVector>& input, bool check_current_state_within_limits = false, bool check_target_state_within_limits = true) const {
        // 1.验证InputParameter的合法性
        // 为什么写template关键字？因为validate是InputParameter的模板成员函数，在这里必须显式告诉编译器这是一个模板函数，否则会报错（编译失败）。
        if (!input.template validate<throw_validation_error>(check_current_state_within_limits, check_target_state_within_limits)) {
            return false;
        }

        // 2.在位置模式下，检查中间点个数
        if (!input.intermediate_positions.empty() && input.control_interface == ControlInterface::Position) {
            if (input.intermediate_positions.size() > max_number_of_waypoints) {
                // 错误处理
                if constexpr (throw_validation_error) {
                    throw RuckigError(std::format("The number of intermediate positions {} exceeds the maximum number of waypoints {}.", input.intermediate_positions.size(), max_number_of_waypoints));
                }
                return false;
            }
        }

        // 3.验证Ruckig实例的delta_time参数
        if (delta_time <= 0.0 && input.duration_discretization != DurationDiscretization::Continuous) {
            if constexpr (throw_validation_error) {
                throw RuckigError(std::format("delta time (control rate) parameter {} should be larger than zero.", delta_time));
            }
            return false;
        }

        return true;
    }

    //! Calculate a new trajectory for the given input
    // 计算新的轨迹--离线计算
    Result calculate(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& trajectory) {
        bool was_interrupted {false};
        return calculate(input, trajectory, was_interrupted);
    }

    //! Calculate a new trajectory for the given input and check for interruption
    Result calculate(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& trajectory, bool& was_interrupted) {
        // 设置check_current_state_within_limits为false,来自传感器,不应该直接拒绝
        // 设置check_target_state_within_limits为true，目标位置必须是“物理可达、合法的”,需要做检查
        if (!validate_input<throw_error>(input, false, true)) {
            return Result::ErrorInvalidInput;
        }

        return calculator.template calculate<throw_error>(input, trajectory, delta_time, was_interrupted);
    }

    //! Get the next output state (with step delta_time) along the calculated trajectory for the given input
    // 在线更新轨迹状态
    Result update(const InputParameter<DOFs, CustomVector>& input, OutputParameter<DOFs, CustomVector>& output) {
        // 1.获取当前时间，用作计算时间消耗
        const auto start = std::chrono::steady_clock::now();
        
        // 2.DOF数目一致性检查（模板 + 运行期双保险）
        if constexpr (DOFs == 0 && throw_error) {
            if (degrees_of_freedom != input.degrees_of_freedom || degrees_of_freedom != output.degrees_of_freedom) {
                throw RuckigError("mismatch in degrees of freedom (vector size).");
            }
        }
        
        // 3.重新规划轨迹标志
        output.new_calculation = false;

        // 4.判断是否需要重新规划
        Result result {Result::Working};
        // current_input_initialized：标志当前输入是否已初始化，默认为false
        // input != current_input：比较当前输入与上次输入是否不同
        if (!current_input_initialized || input != current_input) {
            result = calculate(input, output.trajectory, output.was_calculation_interrupted);
            if (result != Result::Working && result != Result::ErrorPositionalLimits) {
                return result;
            }

            current_input = input;
            current_input_initialized = true;
            output.time = 0.0;
            output.new_calculation = true;
        }

        
        const size_t old_section = output.new_section;
        output.time += delta_time;
        // 轨迹采样
        output.trajectory.at_time(output.time, output.new_position, output.new_velocity, output.new_acceleration, output.new_jerk, output.new_section);
        output.did_section_change = (output.new_section > old_section);  // Report only forward section changes

        // 统计耗时
        const auto stop = std::chrono::steady_clock::now();
        output.calculation_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;

        // current_input重新赋值
        output.pass_to_input(current_input);

        if (output.time > output.trajectory.get_duration()) {
            return Result::Finished;
        }

        return result;
    }
};


template<size_t DOFs, template<class, size_t> class CustomVector = StandardVector>
using RuckigThrow = Ruckig<DOFs, CustomVector, true>;


} // namespace ruckig
