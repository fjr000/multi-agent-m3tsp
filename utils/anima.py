import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import matplotlib.patches as patches
from torch.cuda import graph


def visualize_agent_trajectories(points, trajectories, filename = "anime.gif"):
    """
    可视化多个智能体的轨迹

    参数:
    points: np.ndarray [N,2] - N个点的坐标
    trajectories: np.ndarray [A,T] - A个智能体在T个时间步的轨迹，值为点的序号
    """
    # 设置图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 获取智能体数量和时间步数
    num_agents = trajectories.shape[0]
    num_timesteps = trajectories.shape[1]+1

    # 设置颜色列表，为每个智能体分配不同颜色
    colors = plt.cm.jet(np.linspace(0, 1, num_agents))

    # 绘制所有点
    ax.scatter(points[:, 0], points[:, 1], c='lightgray', s=50)

    # # 为每个点添加编号标签（从1开始）
    # for i, (x, y) in enumerate(points):
    #     ax.text(x, y + 0.01, str(i + 1), ha='center', va='bottom', fontsize=9)

    # 初始化智能体位置标记和轨迹线
    agent_markers = []
    agent_trails = []
    agent_labels = []

    for i in range(num_agents):
        # 创建智能体标记
        marker, = ax.plot([], [], 'o', color=colors[i], markersize=8)
        agent_markers.append(marker)

        # 创建轨迹线
        trail, = ax.plot([], [], '-', color=colors[i], alpha=0.6, linewidth=2)
        agent_trails.append(trail)

        # 创建智能体标签 - 初始化时给定实际的x,y值而不是空列表
        label = ax.text(0, 0, f"Agent {i + 1}", color=colors[i], fontsize=10, ha='center', va='bottom')
        agent_labels.append(label)

    # 设置坐标轴范围
    x_min, x_max = -0.01, 1.01
    y_min, y_max = -0.01, 1.01
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 设置标题和标签
    ax.set_title('智能体轨迹可视化')
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')

    # 添加时间步指示器
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, va='top')

    # 初始化所有箭头对象的列表
    all_arrows = []

    costs = np.zeros((num_agents,))
    costs_depot = np.zeros((num_agents,))


    def init():
        # 初始化函数，清除所有动画元素
        for marker, trail, label in zip(agent_markers, agent_trails, agent_labels):
            marker.set_data([], [])
            trail.set_data([], [])
            # 只设置文本内容，不修改位置
            label.set_text("")
        time_text.set_text('')
        return agent_markers + agent_trails + agent_labels + [time_text]

    def update(frame):
        # 移除上一帧的所有箭头
        for arrow in all_arrows:
            arrow.remove()
        all_arrows.clear()

        # 更新每个智能体的位置和轨迹
        for i, (marker, trail, label) in enumerate(zip(agent_markers, agent_trails, agent_labels)):
            # 获取智能体当前时间步及之前的轨迹点索引
            path_indices = trajectories[i, :frame + 1] - 1  # 减1因为轨迹用1开始的序号表示

            # 获取对应的坐标点
            path_points = points[path_indices]

            # 更新智能体位置标记
            current_point = path_points[-1]
            marker.set_data([current_point[0]], [current_point[1]])

            # 更新智能体标签位置 - 使用set_position方法时需要传入元组而不是列表
            label.set_position((current_point[0], current_point[1] + 0.05))
            if frame == 0:
                costs[i] = 0
                costs_depot[i] = 0
            else:
                cur_pos = path_points[-1]
                last_pos = path_points[-2]
                cost = np.linalg.norm(cur_pos -last_pos).item()
                costs[i] += cost
                costs_depot[i] = costs[i] + np.linalg.norm(path_points[0] - cur_pos).item()
            label.set_text(f"A {i + 1}:{costs[i]:.2f}:{costs_depot[i]:.2f}")

            # 更新轨迹线
            trail.set_data(path_points[:, 0], path_points[:, 1])

            # 添加箭头连接点
            if len(path_points) > 1:
                for j in range(len(path_points) - 1):
                    # 创建从当前点到下一点的箭头
                    start_point = path_points[j]
                    end_point = path_points[j + 1]

                    # 计算箭头方向和长度
                    dx = end_point[0] - start_point[0]
                    dy = end_point[1] - start_point[1]

                    # 创建并添加箭头
                    arrow = ax.arrow(start_point[0], start_point[1], dx, dy,
                                     head_width=0.02, head_length=0.03, fc=colors[i], ec=colors[i],
                                     length_includes_head=True, alpha=0.8)
                    all_arrows.append(arrow)

        max_cost = np.argmax(costs)
        max_cost_depot = np.argmax(costs_depot)
        # 更新时间步显示
        time_text.set_text(f'时间步: {frame + 1}/{num_timesteps}: A:{max_cost+1},{costs[max_cost]:.2f}, A:{max_cost_depot+1},{costs_depot[max_cost_depot]:.2f}')

        return agent_markers + agent_trails + agent_labels + [time_text] + all_arrows

    # 创建动画
    ani = FuncAnimation(fig, update, frames=num_timesteps, init_func=init, blit=True, interval=10)

    # 方法三：保存为GIF (更可靠)
    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=1)
    ani.save(filename, writer=writer)

    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return ani


# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    # 10个点的坐标 [N,2]
    points = np.array([
        [0, 0], [1, 2], [3, 1], [2, 4], [5, 3],
        [4, 0], [6, 2], [7, 4], [8, 1], [9, 3]
    ])

    # 3个智能体在5个时间步的轨迹 [A,T]，值为点的序号（从1开始）
    trajectories = np.array([
        [1, 2, 3, 5, 7],  # 智能体1的轨迹
        [4, 3, 2, 1, 6],  # 智能体2的轨迹
        [7, 8, 9, 10, 5]  # 智能体3的轨迹
    ])

    # 可视化轨迹
    ani = visualize_agent_trajectories(points, trajectories)
