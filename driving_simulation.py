"""
driving_simulation.py  –  Mô phỏng lái xe đơn giản bằng pygame.
Xe chạy trên đường thẳng, khi nhận tín hiệu buồn ngủ sẽ tự lệch lane.
Được điều khiển bởi hệ thống detect fatigue qua shared state.
"""
import pygame
import math
import random
import sys

# ── Cấu hình ──
SCREEN_W, SCREEN_H = 800, 600
FPS = 60
ROAD_W = 300
LANE_W = ROAD_W // 3

# Màu sắc
SKY        = (135, 206, 235)
GRASS      = (34, 139, 34)
ROAD_COLOR = (60, 60, 60)
LINE_WHITE = (255, 255, 255)
LINE_YELLOW = (255, 215, 0)
CAR_BODY   = (0, 100, 200)
CAR_WINDOW = (180, 220, 255)
ALERT_RED  = (220, 30, 30)
ALERT_YELLOW = (255, 200, 0)
HUD_BG     = (0, 0, 0, 160)


class DrivingSimulation:
    """Mô phỏng lái xe – chạy trong thread riêng hoặc cùng main loop."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Driving Simulation - Fatigue Monitor")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("consolas", 36, bold=True)

        # Trạng thái xe
        self.car_x = SCREEN_W // 2         # vị trí ngang
        self.car_target_x = SCREEN_W // 2   # vị trí mục tiêu (center lane)
        self.speed = 80                     # km/h hiển thị
        self.road_offset = 0                # cuộn đường

        # Trạng thái cảnh báo từ hệ thống detect
        self.drowsy_level = 0       # 0=bình thường, 1=cảnh báo nhẹ, 2=nguy hiểm
        self.yawn_count = 0
        self.blink_count = 0
        self.is_monitoring = False
        self.alert_message = ""
        self.alert_timer = 0

        # Drift (lệch lane) khi buồn ngủ
        self.drift = 0.0
        self.drift_target = 0.0
        self.drift_speed = 0.0
        self.swerve_timer = 0

        self.running = True

    def update_fatigue_state(self, drowsy_level=0, yawn_count=0, blink_count=0,
                             is_monitoring=False, alert_msg=""):
        """Cập nhật trạng thái từ hệ thống detect fatigue."""
        self.drowsy_level = drowsy_level
        self.yawn_count = yawn_count
        self.blink_count = blink_count
        self.is_monitoring = is_monitoring
        if alert_msg:
            self.alert_message = alert_msg
            self.alert_timer = 120  # 2 giây ở 60fps

    def _update_drift(self):
        """Xe lệch lane dựa trên mức buồn ngủ."""
        if self.drowsy_level >= 2:
            # Nguy hiểm – drift mạnh, lung lay
            self.swerve_timer += 1
            if self.swerve_timer % 90 == 0:
                self.drift_target = random.uniform(-100, 100)
            self.drift_speed = 1.5
        elif self.drowsy_level == 1:
            # Cảnh báo nhẹ – drift nhẹ
            self.swerve_timer += 1
            if self.swerve_timer % 150 == 0:
                self.drift_target = random.uniform(-40, 40)
            self.drift_speed = 0.6
        else:
            # Bình thường – trở về center
            self.drift_target = 0
            self.drift_speed = 2.0
            self.swerve_timer = 0

        # Smooth drift
        diff = self.drift_target - self.drift
        if abs(diff) > 0.5:
            self.drift += diff * 0.02 * self.drift_speed
        self.car_x = self.car_target_x + self.drift

    def _draw_road(self):
        """Vẽ đường với perspective đơn giản."""
        self.screen.fill(SKY)

        # Cỏ
        pygame.draw.rect(self.screen, GRASS, (0, SCREEN_H // 3, SCREEN_W, SCREEN_H))

        # Đường chính (perspective)
        vanish_y = SCREEN_H // 3
        road_top_w = 60
        road_bot_w = ROAD_W

        for y in range(vanish_y, SCREEN_H):
            t = (y - vanish_y) / (SCREEN_H - vanish_y)  # 0..1
            w = road_top_w + (road_bot_w - road_top_w) * t
            cx = SCREEN_W // 2 + self.drift * t * 0.3
            left = int(cx - w)
            right = int(cx + w)

            # Đường
            pygame.draw.line(self.screen, ROAD_COLOR, (left, y), (right, y))

            # Vạch kẻ đường giữa (nét đứt)
            line_offset = (y + int(self.road_offset * t)) % 40
            if line_offset < 20 and t > 0.05:
                lw = max(1, int(3 * t))
                pygame.draw.line(self.screen, LINE_YELLOW, (int(cx) - lw, y), (int(cx) + lw, y))

            # Vạch lề trái, phải
            if t > 0.05:
                elw = max(1, int(2 * t))
                pygame.draw.line(self.screen, LINE_WHITE, (left, y), (left + elw, y))
                pygame.draw.line(self.screen, LINE_WHITE, (right - elw, y), (right, y))

        # Cuộn đường
        self.road_offset += self.speed * 0.15

    def _draw_car(self):
        """Vẽ xe ở phía dưới màn hình."""
        cx = int(self.car_x)
        cy = SCREEN_H - 80

        # Thân xe
        body = pygame.Rect(cx - 30, cy - 20, 60, 50)
        pygame.draw.rect(self.screen, CAR_BODY, body, border_radius=8)

        # Nóc xe / kính
        roof = pygame.Rect(cx - 22, cy - 40, 44, 25)
        pygame.draw.rect(self.screen, CAR_WINDOW, roof, border_radius=5)

        # Bánh xe
        for dx in [-28, 22]:
            pygame.draw.rect(self.screen, (30, 30, 30), (cx + dx, cy + 22, 12, 8), border_radius=2)

        # Đèn xe
        for dx in [-24, 18]:
            color = ALERT_RED if self.drowsy_level >= 2 else (255, 255, 100)
            pygame.draw.circle(self.screen, color, (cx + dx + 3, cy - 18), 4)

    def _draw_hud(self):
        """Vẽ HUD thông tin."""
        # Panel nền
        hud_surface = pygame.Surface((220, 180), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 160))
        self.screen.blit(hud_surface, (10, 10))

        # Tốc độ
        speed_text = self.font.render(f"Speed: {self.speed} km/h", True, (255, 255, 255))
        self.screen.blit(speed_text, (20, 20))

        # Trạng thái monitoring
        if self.is_monitoring:
            status_color = (0, 255, 0)
            status_text = "MONITORING: ON"
        else:
            status_color = (150, 150, 150)
            status_text = "MONITORING: OFF"
        self.screen.blit(self.font.render(status_text, True, status_color), (20, 50))

        # Drowsy level
        level_colors = [(0, 255, 0), (255, 200, 0), (255, 30, 30)]
        level_names = ["Normal", "Warning", "DANGER!"]
        lv = min(self.drowsy_level, 2)
        self.screen.blit(
            self.font.render(f"State: {level_names[lv]}", True, level_colors[lv]),
            (20, 80)
        )

        # Blink & Yawn
        self.screen.blit(self.font.render(f"Blinks: {self.blink_count}", True, (255, 255, 255)), (20, 110))
        self.screen.blit(self.font.render(f"Yawns:  {self.yawn_count}", True, (255, 255, 255)), (20, 140))

        # Alert message
        if self.alert_timer > 0:
            self.alert_timer -= 1
            # Flash effect
            if self.alert_timer % 20 < 14:
                alert_surf = self.font_big.render(self.alert_message, True, ALERT_RED)
                rect = alert_surf.get_rect(center=(SCREEN_W // 2, 80))
                # Background
                bg = pygame.Surface((rect.width + 20, rect.height + 10), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 180))
                self.screen.blit(bg, (rect.x - 10, rect.y - 5))
                self.screen.blit(alert_surf, rect)

        # Viền đỏ khi nguy hiểm
        if self.drowsy_level >= 2:
            flash = abs(math.sin(pygame.time.get_ticks() * 0.005))
            border_color = (int(220 * flash), 0, 0)
            pygame.draw.rect(self.screen, border_color, (0, 0, SCREEN_W, SCREEN_H), 6)

    def _draw_instructions(self):
        """Vẽ hướng dẫn ở góc phải."""
        instructions = [
            "1 finger: Start detect",
            "2 fingers: Stop detect",
            "3 fingers: Play music",
            "5 fingers: Stop music",
            "Q: Quit",
        ]
        y_start = SCREEN_H - len(instructions) * 25 - 10
        for i, text in enumerate(instructions):
            surf = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(surf, (SCREEN_W - surf.get_width() - 15, y_start + i * 25))

    def tick(self):
        """Gọi mỗi frame – xử lý event, update, render. Trả về False nếu quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.running = False
                return False

        self._update_drift()
        self._draw_road()
        self._draw_car()
        self._draw_hud()
        self._draw_instructions()

        pygame.display.flip()
        self.clock.tick(FPS)
        return True

    def quit(self):
        pygame.quit()


# ── Chạy standalone để test ──
if __name__ == "__main__":
    sim = DrivingSimulation()
    test_timer = 0
    while sim.running:
        test_timer += 1
        # Test: đổi trạng thái mỗi 3 giây
        phase = (test_timer // 180) % 4
        if phase == 0:
            sim.update_fatigue_state(0, 0, 0, True, "")
        elif phase == 1:
            sim.update_fatigue_state(1, 1, 5, True, "DROWSY WARNING!")
        elif phase == 2:
            sim.update_fatigue_state(2, 3, 15, True, "WAKE UP!")
        else:
            sim.update_fatigue_state(0, 0, 0, True, "")

        if not sim.tick():
            break
    sim.quit()
