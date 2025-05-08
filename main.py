import os
import json
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_config import configure

CHINESE_FONT_AVAILABLE = configure()

class MessageType(Enum):
    POSITION = "位置消息"
    VELOCITY = "速度消息"
    IDENTITY = "身份消息"
    EVENT = "状态消息"

@dataclass
class Message:
    time: float
    type: MessageType
    vehicle_id: int
    received: bool = True

class ADSBSimulator:
    def __init__(self, num_vehicles: int, simulation_time: float = 90.0, warmup_time: float = 30.0):
        """
        初始化ADS-B模拟器

        Args:
            num_vehicles: 模拟的发射机数量
            simulation_time: 模拟时间（秒）
            warmup_time: 预热时间（秒）
        """
        self.num_vehicles = num_vehicles
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        self.message_types = [
            (MessageType.POSITION, 0.4, 0.6),
            (MessageType.VELOCITY, 0.4, 0.6),
            (MessageType.IDENTITY, 4.8, 5.2),
            (MessageType.EVENT, 0.4, 0.6)
        ]
        self.message_length = 120e-6  # 消息长度 120us

    def generate_message_times(self) -> List[Message]:
        """为每辆车生成所有类型的消息发送时间"""
        messages = []

        for vehicle_id in range(self.num_vehicles):
            # 为每辆车生成随机初始时间（0-5秒内）
            initial_time = random.uniform(0, 5.0)

            for msg_type, min_interval, max_interval in self.message_types:
                current_time = initial_time

                while current_time <= self.simulation_time:
                    messages.append(Message(current_time, msg_type, vehicle_id))
                    # 生成下一个消息的时间间隔
                    interval = random.uniform(min_interval, max_interval)
                    current_time += interval

        return sorted(messages, key=lambda x: x.time)

    def detect_collisions(self, messages: List[Message]) -> int:
        """检测消息碰撞"""
        failures = 0
        for i in range(len(messages) - 1):
            if messages[i + 1].time - messages[i].time < self.message_length:
                # 标记发生碰撞的消息为未接收
                if messages[i].received:
                    failures += 1
                    messages[i].received = False
                failures += 1
                messages[i + 1].received = False
        return failures

    def calculate_intervals(self, messages: List[Message]) -> Dict[MessageType, List[float]]:
        """统计前后两条成功接收的消息时间间隔"""
        # 按发射机ID和消息类型分组
        vehicle_messages: Dict[int, Dict[MessageType, List[Message]]] = {}

        for msg in messages:
            if msg.vehicle_id not in vehicle_messages:
                vehicle_messages[msg.vehicle_id] = {msg_type: [] for msg_type in MessageType}
            # 只考虑成功接收的消息
            if msg.received:
                vehicle_messages[msg.vehicle_id][msg.type].append(msg)

        # 统计每组消息的前后时间间隔
        msg_intervals = {msg_type: [] for msg_type in MessageType}

        for vehicle_id, msg_dict in vehicle_messages.items():
            for msg_type, msgs in msg_dict.items():
                if len(msgs) < 2:
                    continue
                intervals = [msgs[i + 1].time - msgs[i].time for i in range(len(msgs) - 1)]
                msg_intervals[msg_type].extend(intervals)

        msg_interval_stat = {}
        for msg_type, intervals in msg_intervals.items():
            msg_interval_stat[msg_type] = []
            msg_interval_stat[msg_type].append(np.mean(intervals))
            msg_interval_stat[msg_type].append(min(intervals))
            msg_interval_stat[msg_type].append(max(intervals))
            msg_interval_stat[msg_type].append(np.median(intervals))
            msg_interval_stat[msg_type].append(np.percentile(intervals, 25))
            msg_interval_stat[msg_type].append(np.percentile(intervals, 75))

        return msg_interval_stat

    def run_simulation(self, num_runs: int = 10) -> Tuple[float, Dict[MessageType, List[float]]]:
        """
        运行多次模拟并返回结果

        Args:
            num_runs: 模拟运行次数

        Returns:
            Tuple[float, Dict[MessageType, List[float]]]:
                (平均碰撞概率, 每种消息类型的时间间隔统计)
        """
        failure_probs = []
        msg_interval_stats = []

        for _ in range(num_runs):
            random.seed()
            messages = self.generate_message_times()
            valid_messages = [msg for msg in messages if self.warmup_time <= msg.time <= self.simulation_time]

            failures = self.detect_collisions(valid_messages)
            failure_prob = failures / len(valid_messages)
            failure_probs.append(failure_prob)

            msg_interval_stat = self.calculate_intervals(valid_messages)
            msg_interval_stats.append(msg_interval_stat)

        # 计算平均值
        avg_failure_prob = np.mean(failure_probs)
        avg_msg_interval_stat = {}
        for msg_type in MessageType:
            avg_msg_interval_stat[msg_type] = []
            avg_msg_interval_stat[msg_type].append(np.mean([run[msg_type][0] for run in msg_interval_stats]))
            avg_msg_interval_stat[msg_type].append(min([run[msg_type][1] for run in msg_interval_stats]))
            avg_msg_interval_stat[msg_type].append(max([run[msg_type][2] for run in msg_interval_stats]))
            avg_msg_interval_stat[msg_type].append(np.mean([run[msg_type][3] for run in msg_interval_stats]))
            avg_msg_interval_stat[msg_type].append(np.mean([run[msg_type][4] for run in msg_interval_stats]))
            avg_msg_interval_stat[msg_type].append(np.mean([run[msg_type][5] for run in msg_interval_stats]))

        return avg_failure_prob, avg_msg_interval_stat

def save_simulation_data(vehicle_counts: List[int], avg_failure_probs: List[float],
                        avg_msg_interval_stats: Dict[MessageType, List[List[float]]],
                        filename: str = "simulation_data.json"):
    """保存仿真数据到文件"""
    data = {
        "vehicle_counts": vehicle_counts.tolist(),
        "avg_failure_probs": avg_failure_probs,
        "avg_msg_interval_stats": {
            msg_type.value: stats for msg_type, stats in avg_msg_interval_stats.items()
        }
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def load_simulation_data(filename: str = "simulation_data.json") -> Tuple[List[int], List[float], Dict[MessageType, List[List[float]]]]:
    """从文件加载仿真数据"""
    if not os.path.exists(filename):
        return None, None, None

    with open(filename, "r") as f:
        data = json.load(f)

    vehicle_counts = np.array(data["vehicle_counts"])
    avg_failure_probs = data["avg_failure_probs"]
    avg_msg_interval_stats = {MessageType(msg_type): stats for msg_type, stats in data["avg_msg_interval_stats"].items()}

    return vehicle_counts, avg_failure_probs, avg_msg_interval_stats

def plot_failure_probability(vehicle_counts: List[int], failure_probs: List[float]):
    """绘制消息丢失概率图"""
    plt.figure(figsize=(10, 6))
    # 将概率转换为百分数
    failure_percentages = [prob * 100 for prob in failure_probs]
    plt.plot(vehicle_counts, failure_percentages, "b-", marker="o")
    plt.xlabel("发射机数量")
    plt.ylabel("消息丢失概率 (%)")
    plt.title("ADS-B消息丢失概率与发射机数量的关系")
    plt.grid(True)
    plt.savefig("failure_probability.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_interval_statistics(vehicle_counts: List[int], interval_stats: List[List[float]]):
    """绘制消息时间间隔统计图"""
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(111)
    means, mins, maxes, medians, q1s, q3s = zip(*interval_stats)
    line1 = ax1.plot(vehicle_counts, medians, marker="o", label="中位数")
    line2 = ax1.fill_between(vehicle_counts, q1s, q3s, alpha=0.25, label="第25百分位数到第75百分位数")
    line3 = ax1.plot(vehicle_counts, means, marker="o", label="均值")
    ax1.set_ylabel("消息时间间隔 (秒)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    line4 = ax2.scatter(vehicle_counts, maxes, marker="o", color="r", label="最大值")
    ax2.set_ylabel("最大消息时间间隔 (秒)")

    lines = line1 + [line2] + line3 + [line4]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc="upper left")
    plt.xlabel("发射机数量")
    plt.title("ADS-B消息时间间隔与发射机数量的关系")
    plt.savefig("message_interval.png", dpi=300, bbox_inches="tight")
    plt.close()

def calculate_network_condition(vehicle_counts: List[int], avg_failure_probs: List[float], avg_msg_interval_stats: Dict[MessageType, List[List[float]]]):
    """给定覆盖区域和航空器密度，计算网络情况"""
    civilian_aircraft_density_per_NM2: float = 0.01
    adsb_coverage_radius_km = 300
    adsb_coverage_radius_NM = adsb_coverage_radius_km / 1.852
    adsb_coverage_area_NM2 = 3.14159 * adsb_coverage_radius_NM ** 2
    adsb_coverage_aircraft = adsb_coverage_area_NM2 * civilian_aircraft_density_per_NM2
    print(f"ADS-B覆盖区域飞机数量: {adsb_coverage_aircraft:.2f}")

    avg_failure_prob = np.interp(adsb_coverage_aircraft, vehicle_counts, avg_failure_probs)
    print(f"预计消息冲突概率: {avg_failure_prob:.4f}")

    print("每个消息类型的时间间隔:")
    avg_msg_interval_stat = {}
    for msg_type, stats in avg_msg_interval_stats.items():
        means, mins, maxes, medians, q1s, q3s = zip(*stats)
        stat = [
            np.interp(adsb_coverage_aircraft, vehicle_counts, means),
            np.interp(adsb_coverage_aircraft, vehicle_counts, mins),
            np.interp(adsb_coverage_aircraft, vehicle_counts, maxes),
            np.interp(adsb_coverage_aircraft, vehicle_counts, medians),
            np.interp(adsb_coverage_aircraft, vehicle_counts, q1s),
            np.interp(adsb_coverage_aircraft, vehicle_counts, q3s)
        ]
        avg_msg_interval_stat[msg_type] = stat
        print(f"  {msg_type.value}: 均值={stat[0]:.2f} 最小值={stat[1]:.2f} 最大值={stat[2]:.2f} 中位数={stat[3]:.2f} 第25百分位数={stat[4]:.2f} 第75百分位数={stat[5]:.2f}")

    return avg_failure_prob, avg_msg_interval_stat

def main():
    # 尝试加载已有数据
    vehicle_counts, avg_failure_probs, avg_msg_interval_stats = load_simulation_data()

    # 如果没有数据，则运行仿真
    if vehicle_counts is None:
        # 测试不同发射机数量
        vehicle_counts = np.arange(20, 1001, 20)
        avg_failure_probs = []

        avg_msg_interval_stats = {
            msg_type: [] for msg_type in MessageType
        }

        for num_vehicles in vehicle_counts:
            simulator = ADSBSimulator(num_vehicles)
            avg_failure_prob, avg_msg_interval_stat = simulator.run_simulation()
            avg_failure_probs.append(avg_failure_prob)
            for msg_type in MessageType:
                avg_msg_interval_stats[msg_type].append(avg_msg_interval_stat[msg_type])

            print(f"\n发射机数量: {num_vehicles}")
            print(f"平均碰撞概率: {avg_failure_prob:.4f}")
            print("每个消息类型的时间间隔:")
            for msg_type, stat in avg_msg_interval_stat.items():
                print(f"  {msg_type.value}: 均值={stat[0]:.2f} 最小值={stat[1]:.2f} 最大值={stat[2]:.2f} 中位数={stat[3]:.2f} 第25百分位数={stat[4]:.2f} 第75百分位数={stat[5]:.2f}")

        # 保存仿真数据
        save_simulation_data(vehicle_counts, avg_failure_probs, avg_msg_interval_stats)

    # 绘制图表
    plot_failure_probability(vehicle_counts, avg_failure_probs)
    plot_interval_statistics(vehicle_counts, avg_msg_interval_stats[MessageType.POSITION])

    # 计算网络情况
    calculate_network_condition(vehicle_counts, avg_failure_probs, avg_msg_interval_stats)

if __name__ == "__main__":
    main()