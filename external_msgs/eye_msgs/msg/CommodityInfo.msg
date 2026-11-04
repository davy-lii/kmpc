# ========== 零件识别信息 ==========
# 视野内是否有箱子
std_msgs/Bool has_box           
# 是否防尘帘遮挡 true:有防尘帘遮挡 false：无防尘帘遮挡
std_msgs/Bool has_dusty_shield  
# 手部是否有零件/料箱里是否有零件
std_msgs/Bool has_commodity

# 零件是否可抓取
bool commodity_graspable

# 零件朝向
bool commodity_orientation_correct

# 零件类型/ID（多零件场景）
uint32 commodity_id

# ========== 零件位姿 ==========
# 零件抓取点位姿
geometry_msgs/Pose commodity_pose

# ========== 零件几何信息 ==========
float32 commodity_length               # 长（米）
float32 commodity_width                # 宽（米）
float32 commodity_height               # 高（米）
float32 commodity_weight               # 重量（千克）






