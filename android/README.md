# Vision Agent Android 客户端

手机端采集客户端，无需 USB 调试和 ADB。

## 功能

- **画面采集**：MediaProjection API 截屏（系统弹窗授权，无需 root）
- **控制执行**：AccessibilityService 执行点击/滑动/按键（设置中开启，无需 root）
- **通信协议**：WebSocket，与 PC 客户端 `RemoteCaptureClient` 完全一致
- **自动重连**：网络断开后 5 秒自动重连

## 使用

1. 用 Android Studio 打开 `android/` 目录
2. 构建并安装到手机
3. 输入 Vision Agent 中转服务地址（如 `ws://192.168.1.100:9876`）
4. 点击"开始采集" → 授权截屏
5. （可选）点击"开启无障碍服务" → 在系统设置中启用 Vision Agent → Agent 即可远程控制

## 要求

- Android 7.0+ (API 24)
- 手机与 Vision Agent 主机在同一局域网

## 控制指令

| 指令 | 说明 |
|------|------|
| `{"cmd":"mouse_click", "x":100, "y":200}` | 点击屏幕坐标 |
| `{"cmd":"swipe", "x1":..., "y1":..., "x2":..., "y2":..., "duration":300}` | 滑动 |
| `{"cmd":"key_tap", "key":"back"}` | 返回键 |
| `{"cmd":"key_tap", "key":"home"}` | Home 键 |
| `{"cmd":"key_tap", "key":"recents"}` | 最近任务 |
