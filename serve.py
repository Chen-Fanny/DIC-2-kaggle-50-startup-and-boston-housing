"""
洪水機率預測多元線性迴歸 - 本地互動網頁伺服器
======================================================
執行本程式可於本機啟動 HTTP 伺服器，並自動於預設瀏覽器開啟互動分析平台。
版面與計算機設計參考自 StatsUnlock Multiple Linear Regression Calculator：
https://statsunlock.com/multiple-linear-regression-calculator/
"""

import http.server
import os
import socketserver
import sys
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8000


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    """自訂 HTTP 請求處理器，設定快取控制與 MIME 類型"""

    def end_headers(self):
        # 停用瀏覽器本機快取，確保更新即時反映
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        # 簡化請求紀錄輸出
        sys.stdout.write(f"[HTTP Server] {self.address_string()} - {format % args}\n")


def find_available_port(start_port=DEFAULT_PORT, max_attempts=10):
    """尋找可用通訊埠，若 8000 被佔用則依序遞增"""
    import socket

    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    return start_port


def main():
    # 切換工作目錄至專案根目錄
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    index_file = project_dir / "index.html"
    if not index_file.exists():
        print(f"[錯誤] 找不到 index.html 檔案，請確認其位於: {project_dir}")
        sys.exit(1)

    port = find_available_port(DEFAULT_PORT)
    url = f"http://localhost:{port}/index.html"

    print("=" * 72)
    print("🌐 多元線性迴歸與特徵選擇雙專案互動分析平台 (MLR Explorer)")
    print("   專案 A: 🌊 洪水機率預測 (Flood Probability Prediction)")
    print("   專案 B: 🚀 50 家新創企業淨利潤預測 (50 Startups Profit Prediction)")
    print("📐 版面設計參考: StatsUnlock Multiple Linear Regression Calculator")
    print("   https://statsunlock.com/multiple-linear-regression-calculator/")
    print("=" * 72)
    print(f"🚀 本地伺服器啟動於: {url}")
    print("💡 正在為您自動於預設瀏覽器開啟互動分析儀表板...")
    print("🛑 若欲停止伺服器，請按 Ctrl + C")
    print("=" * 72)

    # 自動開啟瀏覽器
    webbrowser.open(url)

    # 啟動 SocketServer
    with socketserver.TCPServer(("", port), CustomHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[資訊] 伺服器已正常停止。感謝您的使用！")
            httpd.server_close()


if __name__ == "__main__":
    main()
