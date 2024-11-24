import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from scipy.interpolate import splprep, splev


# Hàm khởi tạo lưới
def create_grid():
    grid = np.zeros((16, 16))  # Tạo lưới 16x16
    grid[2:6, 10] = 1  # Thêm chướng ngại vật (obstacle)
    grid[6, 11:13] = 1
    grid[8, 8:11] = 1
    grid[8:11, 8] = 1
    grid[2:5, 3] = 1
    grid[4, 2] = 1
    return grid


# Hàm cập nhật lưới với vùng cảm biến
def update_grid_with_sensing(grid, start, radius):
    sensing_grid = np.full(grid.shape, 3)  # Mặc định tất cả các ô là vùng chưa biết (màu xám)
    rows, cols = grid.shape
    sx, sy = start

    for y in range(rows):
        for x in range(cols):
            distance = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
            if grid[y, x] == 1:  # Giữ nguyên các ô vật cản
                sensing_grid[y, x] = 1
            elif distance <= radius:  # Đổi thành vùng đã biết (màu trắng)
                sensing_grid[y, x] = 2

    return sensing_grid


# Hàm vẽ lưới và đường đi
def plot_grid(grid, start, goal, path=None, smooth_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Lớp màu nền (các ô vuông)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 1:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="red"))  # Chướng ngại vật
            elif grid[y, x] == 2:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="white"))  # Vùng đã biết
            elif grid[y, x] == 3:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="lightslategray"))  # Vùng chưa biết

    # Lớp cạnh ô vuông (đường kẻ)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor="black", linewidth=1))  # Viền đen

    # Hiển thị đường đi ban đầu
    if path:
        path_x, path_y = zip(*[(x + 0.5, y + 0.5) for x, y in path])  # Tọa độ tâm
        ax.plot(path_x, path_y, color="blue", linewidth=2, label="JPS Path")  # Đường đi tìm được
        ax.scatter(path_x, path_y, color="blue", s=50)  # Điểm trên đường đi

    # Hiển thị quỹ đạo cong
    if smooth_path:
        smooth_x, smooth_y = smooth_path
        ax.plot(smooth_x, smooth_y, color="green", linewidth=2, label="Smooth Path")  # Quỹ đạo cong
        # Vẽ các điểm xanh lá cây trên quỹ đạo cong
        for px, py in zip(path_x, path_y):
            nearest_idx = np.argmin((smooth_x - px) ** 2 + (smooth_y - py) ** 2)  # Tìm điểm gần nhất
            ax.scatter(smooth_x[nearest_idx], smooth_y[nearest_idx], color="green", s=50)  # Điểm trên quỹ đạo cong

    # Hiển thị điểm bắt đầu và đích
    ax.add_patch(plt.Rectangle(start, 1, 1, color="cyan", edgecolor="black", linewidth=2, label="Start"))
    ax.add_patch(plt.Rectangle(goal, 1, 1, color="green", edgecolor="black", linewidth=2, label="Goal"))

    #plt.legend()
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.show()


# Hàm tạo quỹ đạo cong, dừng tại vùng xám
def smooth_path_with_limit(path, grid):
    path_x, path_y = zip(*[(x + 0.5, y + 0.5) for x, y in path])  # Tọa độ tâm ô
    tck, _ = splprep([path_x, path_y], s=2)  # Tạo spline nội suy
    smooth_x, smooth_y = splev(np.linspace(0, 1, 100), tck)  # Nội suy quỹ đạo mịn

    # Tìm điểm kết thúc tại vùng xám
    for i, (sx, sy) in enumerate(zip(smooth_x, smooth_y)):
        grid_x, grid_y = int(sx - 0.5), int(sy - 0.5)
        if 0 <= grid_y < grid.shape[0] and 0 <= grid_x < grid.shape[1] and grid[grid_y, grid_x] == 3:
            return smooth_x[:i + 1], smooth_y[:i + 1]  # Dừng lại khi gặp vùng xám

    return smooth_x, smooth_y  # Quỹ đạo đầy đủ nếu không gặp vùng xám


# Hàm Jump Point Search
def jump_point_search(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance heuristic

    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    cost_so_far = {start: 0}

    # Thêm di chuyển đường chéo
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            break

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[1] and 0 <= neighbor[1] < grid.shape[0]:
                if grid[neighbor[1], neighbor[0]] == 1:  # Obstacle
                    continue

                # Cho phép di chuyển chéo nếu không bị cản trở
                if dx != 0 and dy != 0:  # Đường chéo
                    if grid[current[1] + dy, current[0]] == 1 or grid[current[1], current[0] + dx] == 1:
                        continue

                new_cost = cost_so_far[current] + (1.4 if dx != 0 and dy != 0 else 1)  # Chi phí đường chéo
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    open_set.put((priority, neighbor))
                    came_from[neighbor] = current

    # Truy vết đường đi
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current, start)
    path.append(start)
    path.reverse()
    return path


# Main
if __name__ == "__main__":
    grid = create_grid()
    start = (7, 5)  # Điểm bắt đầu
    goal = (15, 4)  # Điểm đích

    # Cập nhật lưới với vùng cảm biến
    sensing_radius = 5
    updated_grid = update_grid_with_sensing(grid, start, sensing_radius)

    # Tìm đường đi
    path = jump_point_search(updated_grid, start, goal)

    # Tạo quỹ đạo cong với giới hạn
    smoothed_path = smooth_path_with_limit(path, updated_grid)

    # Hiển thị kết quả
    plot_grid(updated_grid, start, goal, path, smoothed_path)
