"""Real-time pygame overlays for the unstructured traffic platform."""

from __future__ import annotations

import pygame


class PygameOverlayRenderer:
    """Render hazard overlays and scenario telemetry."""

    def __init__(self, env):
        self.env = env
        self.font_small = None
        self.font_large = None

    def draw(self, agent_surface, sim_surface) -> None:
        if self.font_small is None:
            self.font_small = pygame.font.Font(None, 22)
            self.font_large = pygame.font.Font(None, 28)

        agent_surface.fill((12, 18, 26))
        self._draw_world_overlays(sim_surface)
        self._draw_panel(agent_surface)

    def _draw_world_overlays(self, sim_surface) -> None:
        for pothole in self.env.hazard_manager.potholes:
            x, y = sim_surface.pos2pix(pothole.position[0], pothole.position[1])
            radius = max(4, int(sim_surface.pix(pothole.radius)))
            shade = int(110 + 120 * pothole.severity)
            pygame.draw.circle(sim_surface, (shade, 80, 35), (x, y), radius)
            pygame.draw.circle(sim_surface, (255, 180, 80), (x, y), radius, 1)

        for pedestrian in self.env.hazard_manager.pedestrians:
            x, y = sim_surface.pos2pix(pedestrian.position[0], pedestrian.position[1])
            intensity = int(80 + 175 * pedestrian.intent_probability)
            color = (intensity, 70, 220 - intensity // 2)
            pygame.draw.circle(sim_surface, color, (x, y), 6)
            pygame.draw.circle(sim_surface, (255, 255, 255), (x, y), 9, 1)

        for obstacle in self.env.hazard_manager.static_obstacles:
            x, y = sim_surface.pos2pix(obstacle.position[0], obstacle.position[1])
            pygame.draw.circle(sim_surface, (255, 190, 40), (x, y), 10, 2)

    def _draw_panel(self, surface) -> None:
        metrics = self.env.hazard_manager.metrics(self.env.base_env)
        scenario = self.env.current_scenario

        lines = [
            ("Scenario", scenario.name),
            ("Layout", f"{scenario.layout} | base={scenario.base_env_id}"),
            ("Traffic", f"density={scenario.traffic_density:.2f} | lanes={scenario.lane_count}"),
            ("Hazards", f"obstacles={len(self.env.hazard_manager.static_obstacles)} | potholes={len(self.env.hazard_manager.potholes)} | pedestrians={len(self.env.hazard_manager.pedestrians)}"),
            ("Risk", f"TTC={metrics['same_lane_ttc']:.1f} | scene={metrics['scene_risk']:.2f} | density={metrics['local_density']:.2f}"),
            ("Weather", f"visibility={scenario.visibility:.2f} | friction={scenario.friction:.2f}"),
            ("Driver Models", f"{self.env.driver_library.count} generated profiles active"),
            ("Action Space", "maintain | slow | lane left | lane right | avoid | overtake | defensive"),
        ]

        y = 14
        for idx, (label, text) in enumerate(lines):
            font = self.font_large if idx == 0 else self.font_small
            rendered = font.render(f"{label}: {text}", True, (235, 240, 245))
            surface.blit(rendered, (18, y))
            y += 28 if idx == 0 else 23

        legend_y = max(y + 10, 210)
        surface.blit(self.font_small.render("Driver Aggressiveness", True, (235, 240, 245)), (18, legend_y))
        legend_y += 28

        legend = [
            ("Calm", (80, 210, 110)),
            ("Normal", (240, 210, 70)),
            ("Aggressive", (235, 90, 80)),
        ]
        x = 18
        for label, color in legend:
            pygame.draw.rect(surface, color, (x, legend_y, 22, 14))
            surface.blit(self.font_small.render(label, True, (235, 240, 245)), (x + 30, legend_y - 3))
            x += 145
