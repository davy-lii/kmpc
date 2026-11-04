# ========== 箱子识别信息 ==========
# 有无箱子
std_msgs/Bool has_box

# 箱子是否可抓取
bool box_graspable

# 箱子朝向
bool box_orientation_correct

# 箱子类型/ID（多箱子场景）
uint32 box_id

# ========== 箱子位姿 ==========
# 箱子中心点位姿
geometry_msgs/Pose box_center
# 箱子左侧中心点位姿
geometry_msgs/Pose box_left_center
# 箱子右侧中心点位姿
geometry_msgs/Pose box_right_center

# ========== 箱子几何信息 ==========
float32 box_length               # 长（米）
float32 box_width                # 宽（米）
float32 box_height               # 高（米）
float32 box_weight               # 重量（千克）

# ========== 箱子与机器人的距离 ==========
# 箱子立面到机器人本体Y-Z面最近的距离（单位：米）
float32 box_edge_distance

# 相邻箱子是否与当前箱子接触
bool box_touching

# 推荐拉出方向（选择空间最大的一侧）
uint32 recommended_pull_direction

uint32 PULL_DIRECTION_NONE = 0
uint32 PULL_DIRECTION_LEFT = 1
uint32 PULL_DIRECTION_RIGHT = 2
uint32 PULL_DIRECTION_FRONT = 3
uint32 PULL_DIRECTION_BACK = 4

# 拉出距离（建议拉出多少）
float32 recommended_pull_distance

