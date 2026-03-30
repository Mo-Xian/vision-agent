package com.visionagent.client

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.IBinder
import android.util.DisplayMetrics
import android.view.WindowManager
import androidx.core.app.NotificationCompat

/**
 * 前台服务：使用 MediaProjection 截屏并推送到中转服务。
 *
 * 流程：
 *   1. MainActivity 请求用户授权 MediaProjection
 *   2. 启动本服务，传入授权 resultCode + resultData
 *   3. 服务创建 VirtualDisplay + ImageReader 循环截屏
 *   4. 每帧 JPEG 编码后通过 HubConnection 发送
 */
class CaptureService : Service() {

    companion object {
        const val CHANNEL_ID = "capture_channel"
        const val NOTIFICATION_ID = 1
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        const val EXTRA_SERVER_URL = "server_url"
        const val EXTRA_FPS = "fps"
        const val EXTRA_QUALITY = "quality"

        var instance: CaptureService? = null
            private set
    }

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var hubConnection: HubConnection? = null
    private var captureThread: Thread? = null
    @Volatile private var running = false

    private var screenWidth = 0
    private var screenHeight = 0
    private var screenDpi = 0
    private var fps = 10
    private var quality = 70

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        instance = this
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
            ?: Activity.RESULT_CANCELED
        val resultData = intent?.getParcelableExtra<Intent>(EXTRA_RESULT_DATA)
        val serverUrl = intent?.getStringExtra(EXTRA_SERVER_URL) ?: return START_NOT_STICKY
        fps = intent.getIntExtra(EXTRA_FPS, 10)
        quality = intent.getIntExtra(EXTRA_QUALITY, 70)

        // 前台通知
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Vision Agent 采集中")
            .setContentText("正在推送画面到中转服务")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .build()
        startForeground(NOTIFICATION_ID, notification)

        // 获取屏幕参数
        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)
        screenWidth = metrics.widthPixels
        screenHeight = metrics.heightPixels
        screenDpi = metrics.densityDpi

        // 缩放到合理分辨率（降低带宽）
        val scale = minOf(1.0f, 720.0f / minOf(screenWidth, screenHeight))
        val captureW = (screenWidth * scale).toInt() and 0xFFFE  // 偶数
        val captureH = (screenHeight * scale).toInt() and 0xFFFE

        // 建立 WebSocket 连接
        hubConnection = HubConnection(
            serverUrl = serverUrl,
            fps = fps,
            jpegQuality = quality,
            onLog = { msg -> android.util.Log.i("CaptureService", msg) },
            onStatusChange = {},
            onControl = { cmd -> ControlService.instance?.executeControl(cmd) },
        )
        hubConnection?.start(captureW, captureH)

        // 创建 MediaProjection
        val projectionManager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = projectionManager.getMediaProjection(resultCode, resultData!!)

        // 创建 ImageReader + VirtualDisplay
        imageReader = ImageReader.newInstance(captureW, captureH, PixelFormat.RGBA_8888, 2)
        virtualDisplay = mediaProjection?.createVirtualDisplay(
            "VisionAgent",
            captureW, captureH, screenDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader!!.surface,
            null, null,
        )

        // 截屏循环
        running = true
        captureThread = Thread { captureLoop(captureW, captureH) }.apply {
            isDaemon = true
            start()
        }

        return START_NOT_STICKY
    }

    private fun captureLoop(width: Int, height: Int) {
        val interval = 1000L / fps

        while (running) {
            val t0 = System.currentTimeMillis()

            val image = imageReader?.acquireLatestImage()
            if (image != null) {
                try {
                    val planes = image.planes
                    val buffer = planes[0].buffer
                    val pixelStride = planes[0].pixelStride
                    val rowStride = planes[0].rowStride
                    val rowPadding = rowStride - pixelStride * width

                    val bitmap = Bitmap.createBitmap(
                        width + rowPadding / pixelStride, height,
                        Bitmap.Config.ARGB_8888
                    )
                    bitmap.copyPixelsFromBuffer(buffer)

                    // 裁剪掉 padding
                    val cropped = if (rowPadding > 0) {
                        Bitmap.createBitmap(bitmap, 0, 0, width, height).also {
                            bitmap.recycle()
                        }
                    } else {
                        bitmap
                    }

                    hubConnection?.sendFrame(cropped)
                    cropped.recycle()
                } finally {
                    image.close()
                }
            }

            val elapsed = System.currentTimeMillis() - t0
            val sleepMs = interval - elapsed
            if (sleepMs > 0) {
                Thread.sleep(sleepMs)
            }
        }
    }

    fun stopCapture() {
        running = false
        captureThread?.join(3000)
        virtualDisplay?.release()
        imageReader?.close()
        mediaProjection?.stop()
        hubConnection?.stop()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        instance = null
        running = false
        virtualDisplay?.release()
        imageReader?.close()
        mediaProjection?.stop()
        hubConnection?.stop()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "画面采集",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Vision Agent 画面采集服务"
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }
}
