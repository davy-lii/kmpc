std_msgs/Header header

#动作执行id,增加含义：时间+序号
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


# 欢迎模式（后续）
uint32 EYE_WELCOME_DEMO         = 61
# 回大厅找人（后续）
uint32 EYE_LOBBY_DEMO           = 62

# 任务取消
uint32 EYE_CANCEL               = 100


# 类型
uint32 eye_kind

# 人的名字
string human_name

# 格子编号 1，1
string box_serial_num

