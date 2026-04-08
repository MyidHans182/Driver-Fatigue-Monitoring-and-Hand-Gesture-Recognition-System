"""
driving_simulation.py  –  Mô phỏng lái xe với vehicle model.
- VehicleState: tốc độ, phanh, lái, lane offset, emergency
- Speedometer kim xoay, attention gauge, hazard/brake lights
- Traffic (xe khác), lane departure warning
- Nhận tín hiệu từ Detect_Sleep.py qua update_fatigue_state()
"""
import pygame
import math
import random
from dataclasses import dataclass

# ── Cấu hình ──
SCREEN_W, SCREEN_H = 900, 650
FPS = 60
ROAD_W = 320

# Màu sắc
SKY           = (135, 206, 235)
GRASS         = (34, 139, 34)
ROAD_COLOR    = (60, 60, 60)
LINE_WHITE    = (255, 255, 255)
LINE_YELLOW   = (255, 215, 0)
CAR_BODY      = (0, 100, 200)
CAR_WINDOW    = (180, 220, 255)
ALERT_RED     = (220, 30, 30)
ALERT_YELLOW  = (255, 200, 0)
ALERT_GREEN   = (0, 200, 80)
HAZARD_ORANGE = (255, 140, 0)
GAUGE_BG      = (40, 40, 50)

# Chu kỳ nét đứt đường (pixels)
DASH_CYCLE = 40


# ── VehicleState ──
@dataclass
class VehicleState:
    """Trạng thái xe – dùng để tích hợp mô hình xe thực."""
    speed: float = 0.0
    max_speed: float = 80.0
    target_speed: float = 80.0
    steering_angle: float = 0.0
    brake_force: float = 0.0
    throttle: float = 1.0
    lane_offset: float = 0.0
    is_emergency: bool = False
    hazard_on: bool = False
    brake_light: bool = False
    lane_departure: bool = False

    def to_dict(self):
        """Export trạng thái xe dưới dạng dict (cho logging/CAN bus)."""
        return {
            "speed_kmh": round(self.speed, 1),
            "target_speed_kmh": round(self.target_speed, 1),
            "steering_angle": round(self.steering_angle, 3),
            "brake_force": round(self.brake_force, 2),
            "throttle": round(self.throttle, 2),
            "lane_offset_px": round(self.lane_offset, 1),
            "is_emergency": self.is_emergency,
            "hazard": self.hazard_on,
            "brake_light": self.brake_light,
            "lane_departure": self.lane_departure,
        }


# ── Traffic Car ──
@dataclass
class TrafficCar:
    x: float = 0.0
    y: float = 0.0
    speed: float = 60.0
    color: tuple = (180, 30, 30)
    lane: int = 0  # -1 = trái, 1 = phải


class DrivingSimulation:
    """Mô phỏng lái xe – chạy cùng main loop của Detect_Sleep."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Driving Simulation - Fatigue Monitor")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_big = pygame.font.SysFont("consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 14)

        # Vehicle state
        self.vehicle = VehicleState()
        self.vehicle.speed = self.vehicle.max_speed

        # Vị trí xe trên màn hình
        self.car_screen_x = SCREEN_W // 2
        self.road_offset = 0.0

        # Fatigue data từ detection system
        self.drowsy_level = 0
        self.attention_score = 100.0
        self.yawn_count = 0
        self.blink_count = 0
        self.is_monitoring = False
        self.alert_message = ""
        self.alert_timer = 0

        # Head pose data
        self.head_pitch = 0.0
        self.head_yaw = 0.0
        self.head_roll = 0.0

        # Drift
        self.drift = 0.0
        self.drift_target = 0.0
        self.drift_speed = 0.0
        self.swerve_timer = 0

        # Traffic
        self.traffic: list[TrafficCar] = []
        self.traffic_spawn_timer = 0

        # Animation
        self.frame_count = 0
        self.running = True

    # ── API: nhận data từ Detect_Sleep ──
    def update_fatigue_state(self, drowsy_level=0, yawn_count=0, blink_count=0,
                             is_monitoring=False, alert_msg="",
                             attention_score=100.0,
                             head_pitch=0.0, head_yaw=0.0, head_roll=0.0):
        """Cập nhật trạng thái từ hệ thống detect fatigue."""
        self.drowsy_level = drowsy_level
        self.yawn_count = yawn_count
        self.blink_count = blink_count
        self.is_monitoring = is_monitoring
        self.attention_score = attention_score
        self.head_pitch = head_pitch
        self.head_yaw = head_yaw
        self.head_roll = head_roll
        if alert_msg:
            self.alert_message = alert_msg
            self.alert_timer = 120

    def get_vehicle_state(self) -> dict:
        """Lấy trạng thái xe hiện tại (cho logging hoặc CAN bus)."""
        return self.vehicle.to_dict()

    # ── Vehicle Control Logic ──
    def _update_vehicle(self):
        """Điều khiển xe dựa trên attention_score."""
        score = self.attention_score
        v = self.vehicle

        if not self.is_monitoring:
            v.target_speed = v.max_speed
            v.brake_force = 0.0
            v.throttle = 1.0
            v.is_emergency = False
            v.hazard_on = False
            v.brake_light = False
        elif score > 80:
            v.target_speed = v.max_speed
            v.brake_force = 0.0
            v.throttle = 1.0
            v.is_emergency = False
            v.hazard_on = False
            v.brake_light = False
        elif score > 60:
            v.target_speed = v.max_speed * 0.9
            v.brake_force = 0.1
            v.throttle = 0.8
            v.is_emergency = False
            v.hazard_on = False
            v.brake_light = True
        elif score > 40:
            v.target_speed = v.max_speed * 0.7
            v.brake_force = 0.3
            v.throttle = 0.5
            v.is_emergency = False
            v.hazard_on = True
            v.brake_light = True
        elif score > 20:
            v.target_speed = v.max_speed * 0.3
            v.brake_force = 0.6
            v.throttle = 0.0
            v.is_emergency = True
            v.hazard_on = True
            v.brake_light = True
        else:
            v.target_speed = 0
            v.brake_force = 1.0
            v.throttle = 0.0
            v.is_emergency = True
            v.hazard_on = True
            v.brake_light = True

        # Smooth speed transition
        speed_diff = v.target_speed - v.speed
        if abs(speed_diff) > 0.5:
            accel = 0.3 if speed_diff > 0 else -0.8
            if v.brake_force > 0.5:
                accel = -1.5
            v.speed += accel
        v.speed = max(0, min(v.speed, v.max_speed))

        # Lane departure detection
        v.lane_offset = self.drift
        v.lane_departure = abs(self.drift) > 60

    def _update_drift(self):
        """Xe lệch lane dựa trên mức buồn ngủ + head yaw."""
        head_drift = self.head_yaw * 0.5 if self.is_monitoring else 0

        if self.drowsy_level >= 2:
            self.swerve_timer += 1
            if self.swerve_timer % 70 == 0:
                self.drift_target = random.uniform(-120, 120) + head_drift
            self.drift_speed = 2.0
        elif self.drowsy_level == 1:
            self.swerve_timer += 1
            if self.swerve_timer % 120 == 0:
                self.drift_target = random.uniform(-50, 50) + head_drift
            self.drift_speed = 0.8
        else:
            self.drift_target = head_drift * 0.2
            self.drift_speed = 2.5
            self.swerve_timer = 0

        diff = self.drift_target - self.drift
        if abs(diff) > 0.3:
            self.drift += diff * 0.025 * self.drift_speed
        self.car_screen_x = SCREEN_W // 2 + self.drift

        self.vehicle.steering_angle = max(-1, min(1, self.drift / 120.0))

    def _update_traffic(self):
        """Spawn và update xe traffic."""
        self.traffic_spawn_timer += 1
        if self.traffic_spawn_timer > 180 and len(self.traffic) < 3:
            lane = random.choice([-1, 1])
            color = random.choice([
                (180, 30, 30), (30, 130, 30), (160, 160, 30), (200, 200, 200)
            ])
            car = TrafficCar(
                x=SCREEN_W // 2 + lane * 80,
                y=SCREEN_H // 3 - 20,
                speed=random.uniform(40, 70),
                color=color,
                lane=lane,
            )
            self.traffic.append(car)
            self.traffic_spawn_timer = 0

        vanish_y = SCREEN_H // 3
        road_h = SCREEN_H - vanish_y
        for car in self.traffic:
            relative_speed = self.vehicle.speed - car.speed
            car.y += relative_speed * 0.05 + 1.5
            t = max(0, (car.y - vanish_y)) / road_h
            car.x = SCREEN_W // 2 + car.lane * (30 + 60 * t) + self.drift * t * 0.3

        self.traffic = [c for c in self.traffic if c.y < SCREEN_H + 50]

    # ── Drawing ──
    def _draw_road(self):
        """Vẽ đường với perspective – polygon cho thân đường, loop chỉ cho vạch."""
        self.screen.fill(SKY)

        vanish_y = SCREEN_H // 3
        road_top_w = 60
        road_bot_w = ROAD_W

        # Cỏ (1 rect)
        pygame.draw.rect(self.screen, GRASS, (0, vanish_y, SCREEN_W, SCREEN_H - vanish_y))

        # Đường chính – 1 polygon thay vì pixel-by-pixel
        cx_top = SCREEN_W // 2 + self.drift * 0.0  # vanishing point không drift
        cx_bot = SCREEN_W // 2 + self.drift * 0.3
        road_poly = [
            (int(cx_top - road_top_w), vanish_y),
            (int(cx_top + road_top_w), vanish_y),
            (int(cx_bot + road_bot_w), SCREEN_H),
            (int(cx_bot - road_bot_w), SCREEN_H),
        ]
        pygame.draw.polygon(self.screen, ROAD_COLOR, road_poly)

        # Vạch lề trái + phải (2 polylines)
        edge_pts_left = []
        edge_pts_right = []
        for y in range(vanish_y, SCREEN_H, 4):  # mỗi 4px, đủ mượt
            t = (y - vanish_y) / (SCREEN_H - vanish_y)
            w = road_top_w + (road_bot_w - road_top_w) * t
            cx = SCREEN_W // 2 + self.drift * t * 0.3
            edge_pts_left.append((int(cx - w), y))
            edge_pts_right.append((int(cx + w), y))

        if len(edge_pts_left) > 1:
            pygame.draw.lines(self.screen, LINE_WHITE, False, edge_pts_left, 2)
            pygame.draw.lines(self.screen, LINE_WHITE, False, edge_pts_right, 2)

        # Vạch giữa (nét đứt vàng) – chỉ vẽ đoạn nét, không vẽ khoảng trống
        for y in range(vanish_y + 10, SCREEN_H, 4):
            t = (y - vanish_y) / (SCREEN_H - vanish_y)
            cx = SCREEN_W // 2 + self.drift * t * 0.3
            line_offset = (y + int(self.road_offset * t)) % DASH_CYCLE
            if line_offset < DASH_CYCLE // 2:
                lw = max(1, int(3 * t))
                pygame.draw.line(self.screen, LINE_YELLOW,
                                 (int(cx) - lw, y), (int(cx) + lw, y))

        # Cuộn đường theo speed – wrap around để tránh overflow
        self.road_offset = (self.road_offset + self.vehicle.speed * 0.15) % DASH_CYCLE

    def _draw_traffic(self):
        """Vẽ xe traffic."""
        vanish_y = SCREEN_H // 3
        road_h = SCREEN_H - vanish_y
        for car in self.traffic:
            if car.y < vanish_y:
                continue
            t = (car.y - vanish_y) / road_h
            scale = 0.3 + 0.7 * t
            w = int(30 * scale)
            h = int(20 * scale)
            rect = pygame.Rect(int(car.x) - w // 2, int(car.y) - h // 2, w, h)
            pygame.draw.rect(self.screen, car.color, rect,
                             border_radius=max(1, int(4 * scale)))
            # Kính xe
            rw = int(20 * scale)
            rh = int(8 * scale)
            roof = pygame.Rect(int(car.x) - rw // 2, int(car.y) - h // 2 - rh, rw, rh)
            pygame.draw.rect(self.screen, (100, 150, 200),
                             roof, border_radius=max(1, int(3 * scale)))

    def _draw_car(self):
        """Vẽ xe player."""
        cx = int(self.car_screen_x)
        cy = SCREEN_H - 80

        # Thân xe
        body = pygame.Rect(cx - 30, cy - 20, 60, 50)
        pygame.draw.rect(self.screen, CAR_BODY, body, border_radius=8)

        # Nóc xe
        roof = pygame.Rect(cx - 22, cy - 40, 44, 25)
        pygame.draw.rect(self.screen, CAR_WINDOW, roof, border_radius=5)

        # Bánh xe
        for dx in [-28, 22]:
            pygame.draw.rect(self.screen, (30, 30, 30),
                             (cx + dx, cy + 22, 12, 8), border_radius=2)

        # Đèn trước
        for dx in [-24, 18]:
            pygame.draw.circle(self.screen, (255, 255, 100), (cx + dx + 3, cy - 18), 4)

        # Đèn phanh
        if self.vehicle.brake_light:
            flash = abs(math.sin(self.frame_count * 0.1))
            for dx in [-22, 20]:
                pygame.draw.circle(self.screen, (int(255 * flash), 0, 0),
                                   (cx + dx, cy + 28), 4)

        # Hazard lights
        if self.vehicle.hazard_on:
            flash = (self.frame_count // 20) % 2 == 0
            if flash:
                for dx in [-32, 30]:
                    pygame.draw.circle(self.screen, HAZARD_ORANGE,
                                       (cx + dx, cy), 5)

    def _draw_speedometer(self):
        """Vẽ đồng hồ tốc độ (kim xoay)."""
        center_x, center_y = 100, SCREEN_H - 90
        radius = 55

        pygame.draw.circle(self.screen, GAUGE_BG, (center_x, center_y), radius)
        pygame.draw.circle(self.screen, (80, 80, 90), (center_x, center_y), radius, 3)

        for i in range(0, 9):
            angle = math.radians(225 - i * 270 / 8)
            x1 = center_x + int((radius - 12) * math.cos(angle))
            y1 = center_y - int((radius - 12) * math.sin(angle))
            x2 = center_x + int((radius - 4) * math.cos(angle))
            y2 = center_y - int((radius - 4) * math.sin(angle))
            pygame.draw.line(self.screen, LINE_WHITE, (x1, y1), (x2, y2), 2)

            speed_val = i * 10
            txt = self.font_small.render(str(speed_val), True, (180, 180, 180))
            tx = center_x + int((radius - 22) * math.cos(angle)) - txt.get_width() // 2
            ty = center_y - int((radius - 22) * math.sin(angle)) - txt.get_height() // 2
            self.screen.blit(txt, (tx, ty))

        speed_ratio = min(self.vehicle.speed / 80.0, 1.0)
        needle_angle = math.radians(225 - speed_ratio * 270)
        nx = center_x + int((radius - 18) * math.cos(needle_angle))
        ny = center_y - int((radius - 18) * math.sin(needle_angle))

        needle_color = ALERT_GREEN
        if self.vehicle.speed < 30:
            needle_color = ALERT_RED if self.vehicle.is_emergency else ALERT_YELLOW
        pygame.draw.line(self.screen, needle_color, (center_x, center_y), (nx, ny), 3)
        pygame.draw.circle(self.screen, LINE_WHITE, (center_x, center_y), 5)

        speed_txt = self.font.render(f"{int(self.vehicle.speed)} km/h", True, LINE_WHITE)
        self.screen.blit(speed_txt,
                         (center_x - speed_txt.get_width() // 2, center_y + 20))

    def _draw_attention_gauge(self):
        """Vẽ thanh attention score."""
        x, y = SCREEN_W - 60, SCREEN_H - 180
        w, h = 30, 150

        pygame.draw.rect(self.screen, GAUGE_BG, (x - 2, y - 2, w + 4, h + 4),
                         border_radius=4)

        fill_h = int(h * self.attention_score / 100.0)
        if self.attention_score > 60:
            color = ALERT_GREEN
        elif self.attention_score > 30:
            color = ALERT_YELLOW
        else:
            color = ALERT_RED

        if fill_h > 0:
            fill_rect = pygame.Rect(x, y + (h - fill_h), w, fill_h)
            pygame.draw.rect(self.screen, color, fill_rect, border_radius=2)

        pygame.draw.rect(self.screen, (100, 100, 110), (x, y, w, h), 2, border_radius=4)

        label = self.font_small.render("ATT", True, LINE_WHITE)
        self.screen.blit(label, (x + w // 2 - label.get_width() // 2, y + h + 5))
        score_txt = self.font.render(f"{int(self.attention_score)}", True, color)
        self.screen.blit(score_txt,
                         (x + w // 2 - score_txt.get_width() // 2, y - 25))

    def _draw_head_pose_indicator(self):
        """Vẽ indicator hướng đầu (hình tròn + mũi tên)."""
        cx, cy = SCREEN_W - 100, 80
        r = 35

        bg_surf = pygame.Surface((r * 2 + 10, r * 2 + 10), pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 120))
        self.screen.blit(bg_surf, (cx - r - 5, cy - r - 5))

        pygame.draw.circle(self.screen, (180, 180, 180), (cx, cy), r, 2)

        arrow_x = cx + int(self.head_yaw * 0.8)
        arrow_y = cy + int(self.head_pitch * 0.8)
        arrow_x = max(cx - r + 5, min(cx + r - 5, arrow_x))
        arrow_y = max(cy - r + 5, min(cy + r - 5, arrow_y))

        dot_color = ALERT_GREEN
        if abs(self.head_yaw) > 25 or abs(self.head_pitch) > 20:
            dot_color = ALERT_RED
        elif abs(self.head_yaw) > 15 or abs(self.head_pitch) > 10:
            dot_color = ALERT_YELLOW
        pygame.draw.circle(self.screen, dot_color, (arrow_x, arrow_y), 6)
        pygame.draw.line(self.screen, dot_color, (cx, cy), (arrow_x, arrow_y), 2)

        label = self.font_small.render("HEAD", True, LINE_WHITE)
        self.screen.blit(label, (cx - label.get_width() // 2, cy + r + 8))

        roll_txt = self.font_small.render(f"R:{int(self.head_roll)}\u00b0", True, (180, 180, 180))
        self.screen.blit(roll_txt, (cx - roll_txt.get_width() // 2, cy - r - 18))

    def _draw_hud(self):
        """Vẽ HUD thông tin."""
        hud_surface = pygame.Surface((240, 200), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 160))
        self.screen.blit(hud_surface, (10, 10))

        y_pos = 20

        if self.is_monitoring:
            status_color = ALERT_GREEN
            status_text = "MONITORING: ON"
        else:
            status_color = (150, 150, 150)
            status_text = "MONITORING: OFF"
        self.screen.blit(self.font.render(status_text, True, status_color), (20, y_pos))
        y_pos += 30

        level_colors = [ALERT_GREEN, ALERT_YELLOW, ALERT_RED]
        level_names = ["Normal", "Warning", "DANGER!"]
        lv = min(self.drowsy_level, 2)
        self.screen.blit(
            self.font.render(f"State: {level_names[lv]}", True, level_colors[lv]),
            (20, y_pos)
        )
        y_pos += 30

        att_color = ALERT_GREEN if self.attention_score > 60 else (
            ALERT_YELLOW if self.attention_score > 30 else ALERT_RED)
        self.screen.blit(
            self.font.render(f"Attention: {int(self.attention_score)}%", True, att_color),
            (20, y_pos)
        )
        y_pos += 30

        self.screen.blit(
            self.font.render(f"Blinks: {self.blink_count}", True, LINE_WHITE),
            (20, y_pos)
        )
        y_pos += 25
        self.screen.blit(
            self.font.render(f"Yawns:  {self.yawn_count}", True, LINE_WHITE),
            (20, y_pos)
        )
        y_pos += 25

        v = self.vehicle
        brake_txt = f"Brake: {int(v.brake_force * 100)}%"
        brake_color = ALERT_RED if v.brake_force > 0.5 else (
            ALERT_YELLOW if v.brake_force > 0 else LINE_WHITE)
        self.screen.blit(self.font.render(brake_txt, True, brake_color), (20, y_pos))

        # Alert message
        if self.alert_timer > 0:
            self.alert_timer -= 1
            if self.alert_timer % 20 < 14:
                alert_surf = self.font_big.render(self.alert_message, True, ALERT_RED)
                rect = alert_surf.get_rect(center=(SCREEN_W // 2, 40))
                bg = pygame.Surface((rect.width + 20, rect.height + 10), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 200))
                self.screen.blit(bg, (rect.x - 10, rect.y - 5))
                self.screen.blit(alert_surf, rect)

        # Viền đỏ khi nguy hiểm
        if self.drowsy_level >= 2:
            flash = abs(math.sin(self.frame_count * 0.08))
            border_color = (int(220 * flash), 0, 0)
            pygame.draw.rect(self.screen, border_color, (0, 0, SCREEN_W, SCREEN_H), 6)

        # Lane departure warning
        if self.vehicle.lane_departure:
            flash = (self.frame_count // 10) % 2 == 0
            if flash:
                pygame.draw.rect(self.screen, ALERT_YELLOW,
                                 (0, 0, SCREEN_W, SCREEN_H), 4)

    def _draw_instructions(self):
        """Hướng dẫn góc phải dưới."""
        instructions = [
            "1 finger: Start detect",
            "2 fingers: Stop detect",
            "3 fingers: Play music",
            "5 fingers: Stop music",
            "Q: Quit",
        ]
        y_start = SCREEN_H - len(instructions) * 22 - 10
        for i, text in enumerate(instructions):
            surf = self.font_small.render(text, True, (160, 160, 160))
            self.screen.blit(surf, (SCREEN_W - surf.get_width() - 15,
                                    y_start + i * 22))

    # ── Main tick ──
    def tick(self):
        """Gọi mỗi frame. Trả về False nếu quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.running = False
                return False

        self.frame_count += 1
        self._update_drift()
        self._update_vehicle()
        self._update_traffic()

        self._draw_road()
        self._draw_traffic()
        self._draw_car()
        self._draw_hud()
        self._draw_speedometer()
        self._draw_attention_gauge()
        self._draw_head_pose_indicator()
        self._draw_instructions()

        pygame.display.flip()
        self.clock.tick(FPS)
        return True

    def quit(self):
        pygame.quit()


# ── Standalone test ──
if __name__ == "__main__":
    sim = DrivingSimulation()
    test_timer = 0
    while sim.running:
        test_timer += 1
        phase = (test_timer // 180) % 5
        if phase == 0:
            sim.update_fatigue_state(0, 0, 0, True, "", 95.0, 0, 0, 0)
        elif phase == 1:
            sim.update_fatigue_state(1, 1, 5, True, "DROWSY!", 55.0, -10, 5, 3)
        elif phase == 2:
            sim.update_fatigue_state(2, 3, 15, True, "WAKE UP!", 25.0, -20, 15, 8)
        elif phase == 3:
            sim.update_fatigue_state(2, 4, 20, True, "EMERGENCY!", 10.0, -25, 20, 12)
        else:
            sim.update_fatigue_state(0, 0, 0, True, "Recovered", 90.0, 0, 0, 0)

        if not sim.tick():
            break
    sim.quit()
