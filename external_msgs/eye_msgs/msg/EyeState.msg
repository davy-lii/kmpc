std_msgs/Header header

# ActionState
uint8 ACTION_IDLE = 0
uint8 ACTION_RUNNING = 1
uint8 ACTION_DONE = 2

# ActionResult
uint8 ACTION_NONE = 0
uint8 ACTION_SUCCESS = 1
uint8 ACTION_FAIL = 2

# ErrCode
uint16 ACTION_ERR_NONE = 0
uint16 ACTION_ERR_NOWORK = 1

# 当前执行的id
uint64 action_id

# 人脸录入
uint32 EYE_FACE_RECORD          = 5
# 身体录入
uint32 EYE_BODY_RECORD          = 6  

# 料箱货物确认
uint32 COMMODITY_STAUTS_CONFIRM = 7
# 置物架确认
uint32 COMMODITY_SHELF_CONFIRM  = 8    
# 抓取状态确认
uint32 GRAB_STATUS_CONFIRM      = 9   
# 箱子状态确认
uint32 BOX_STATUS_CONFIRM       = 10
# 放置箱子位置确认
uint32 BOX_PLACE_POS_CONFIRM    = 11
# 放置零件状态确认
uint32 PLACE_STATUS_CONFIRM    = 12


# 欢迎模式
uint32 EYE_WELCOME_DEMO         = 61
# 回大厅找人
uint32 EYE_LOBBY_DEMO           = 62

# 任务取消（不需要请求）
uint32 EYE_CANCEL               = 100

# 架子料孔位置
geometry_msgs/Pose hole_pose

# 料孔是否为空
bool hole_is_empty

# 零件是否在料孔内放好
bool commodities_is_well_seated

# 零件信息
eye_msgs/CommodityInfo[] commodities_info

# 检测到的零件总数
uint32 total_commodities

# 箱子信息
eye_msgs/BoxInfo box_info

# 类型
uint32 eye_kind

# 作业类型
uint32 job_kind
# 0: NONE
# 1: 全身回正
# 2: 抓取零件
# 3: 放置零件
# 4: 搬箱子
# 5: 放箱子
# 6: 呼叫AGV

# 检测到的目标类型(预留字段)
uint32 target_type

# 检测结果类型
uint32 DETECT_NONE              = 0
uint32 DETECT_COMMODITY         = 1   # 检测到零件
uint32 DETECT_SHELF             = 2   # 检测到料架
uint32 DETECT_BOX               = 3   # 检测到箱子

# 执行状态：空闲/正在执行/执行完成
uint8 exe_state
# 执行结果：无效/成功/失败
uint8 exe_result

# 错误码
uint16 err_code
# 错误消息
string err_info
# 剩余执行时间
uint64 wait_for_millisec
