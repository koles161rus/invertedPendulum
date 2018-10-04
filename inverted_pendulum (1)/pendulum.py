from math import *
from numpy import *
import pygame
import numpy as np
import control.matlab
from main import InvertedPendulumBalancer


dt = 0.01
g = 9.81
l = 1.0
m = 1.0

global k1, k2, k3
k1 = 500000
k2 = 10000
k3 = 10000

global q1, q2, q3, q4, r
q1 = 0.001
q2 = 0.001
q3 = 1000000000
q4 = 20000000
r = 0.005

clock = pygame.time.Clock()
pygame.init()
size = (width, height) = (1800, 1000)
screen = pygame.display.set_mode(size)


class Pendulum:
    def __init__(self, x0, y0, phi0):
        self.phi0 = phi0
        self.phi = phi0
        self.velocity = 0
        self.x0 = x0
        self.y0 = y0
        self.x0_vel = 0
        self.x = x0 + 250.0 * sin(phi0)
        self.y = y0 + 250.0 * cos(phi0)

        self.t = dt
        self.t = np.arange(0, 30, 0.01)

        self.phi_chart_t = 20
        self.phi_chart = [(self.phi_chart_t, 820)]

        self.x_chart_t = 20
        self.x_chart = [(self.x_chart_t, 480)]

    def move(self, control):
        self.phi = atan2(self.x - self.x0, self.y - self.y0)
        d_velocity = -g * sin(self.phi) * dt / l
        self.velocity += d_velocity
        d_phi = dt * self.velocity
        self.phi += d_phi
        self.x = self.x0 + 250.0 * sin(self.phi)
        self.y = self.y0 + 250.0 * cos(self.phi)

        d_x0_vel = dt * control
        self.x0_vel += d_x0_vel
        dx0 = dt * self.x0_vel
        self.x0 += dx0

    def draw(self):
        pygame.draw.circle(screen, (0, 0, 0), [int(self.x0), int(self.y0)], 5)
        pygame.draw.line(screen, (0, 0, 0), [self.x0, self.y0], [self.x, self.y], 2)
        pygame.draw.circle(screen, (255, 0, 0), [int(self.x), int(self.y)], 10)
        pygame.draw.line(screen, (0, 0, 0), [0, self.y0], [1800, self.y0], 3)

        self.phi_chart_t += 0.2
        if self.phi_chart_t > size[0]:
            self.phi_chart_t = 0
            self.phi_chart = [(self.phi_chart_t, 820)]
        angle = np.pi - self.phi if self.phi > 0 else -np.pi - self.phi
        self.phi_chart.append((self.phi_chart_t, 300 * angle + 820))
        pygame.draw.lines(screen, (255, 0, 0), False, self.phi_chart, 3)
        pygame.draw.line(screen, (0, 0, 0), [20, 820], [1780, 820], 2)
        pygame.draw.line(screen, (0, 0, 0), [20, 660], [20, 980], 2)
        pygame.draw.line(screen, (128, 128, 128), [20, 665], [1780, 665], 2)
        pygame.draw.line(screen, (128, 128, 128), [20, 975], [1780, 975], 2)
        pygame.draw.polygon(screen, (0, 0, 0), ((18, 660), (20, 650), (22, 660)), 2)
        pygame.draw.polygon(screen, (0, 0, 0), ((1770, 822), (1780, 820), (1770, 818)), 2)
        print(self.phi)

        self.x_chart_t += 0.2
        if self.x_chart_t > size[0]:
            self.x_chart_t = 0
            self.x = [(self.x_chart_t, 480)]
        move = self.x
        self.x_chart.append((self.x_chart_t, -0.2 * move + 683))
        pygame.draw.lines(screen, (0, 255, 0), False, self.x_chart, 3)
        pygame.draw.line(screen, (0, 0, 0), [20, 480], [1780, 480], 2)
        pygame.draw.line(screen, (0, 0, 0), [20, 620], [20, 340], 2)
        pygame.draw.line(screen, (128, 128, 128), [20, 345], [1780, 345], 2)
        pygame.draw.line(screen, (128, 128, 128), [20, 615], [1780, 615], 2)
        pygame.draw.polygon(screen, (0, 0, 0), ((18, 340), (20, 330), (22, 340)), 2)
        pygame.draw.polygon(screen, (0, 0, 0), ((1770, 482), (1780, 480), (1770, 478)), 2)
        print(self.x)


class PID:
    def __init__(self, k1, k2, k3, pendulum):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p = pendulum
        self.error = pi - self.p.phi
        self.derivative = 0
        self.integral = 0

    def update(self):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        tmp = self.error
        self.error = copysign(1, p.phi) * (pi - abs(self.p.phi)) + (self.p.x0 - 600) / 10000
        diff = self.error - tmp
        self.derivative = diff / dt
        self.integral += tmp

    def output(self):
        return self.k1 * self.error + self.k2 * self.derivative + self.k3 * self.integral


class LQR:
    def __init__(self, q1, q2, q3, q4, r, pendulum):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.p = pendulum
        self.A = matrix([[0, 1, 0, 0], [0, 0, -g, 0], [0, 0, 0, 1], [0, 0, 2 * g, 0]])
        self.B = matrix([[0], [1], [0], [-1]])
        self.Q = diag([q1, q2, q3, q4])
        self.R = r
        self.K = control.matlab.lqr(self.A, self.B, self.Q, self.R)[0]
        print(self.K)

    def update(self):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.Q = diag([q1, q2, q3, q4])
        self.K = control.matlab.lqr(self.A, self.B, self.Q, self.R)[0]

    def output(self):
        X = matrix([[-(self.p.x0 - 600) / 10], [self.p.x0_vel / 10],
                    [copysign(1, self.p.phi) * (-pi + abs(self.p.phi))], [self.p.velocity]])
        U = self.K * X
        return U.flat[0]


class DrawPID:
    def draw_text(self):
        myfont = pygame.font.SysFont("monospace", 15)
        label1 = myfont.render("Пропорциональный коэффициент: %d" % k1, 1, (255, 0, 0))
        screen.blit(label1, (100, 400))
        label2 = myfont.render("Дифференциальный коэффициент: %d" % k2, 1, (255, 0, 0))
        screen.blit(label2, (100, 420))
        label3 = myfont.render("Интегральный коэффициент: %.1f" % k3, 1, (255, 0, 0))
        screen.blit(label3, (100, 440))


    def make_buttons(k1, k2, k3, pid, pend):
        pygame.draw.rect(screen, (0, 0, 255), [420, 400, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [520, 400, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [420, 420, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [520, 420, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [420, 440, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [520, 440, 90, 15])

        myfont = pygame.font.SysFont("monospace", 15)
        label1 = myfont.render("Повысить", 1, (255, 255, 255))
        label2 = myfont.render("Понизить", 1, (255, 255, 255))
        screen.blit(label1, (420, 400))
        screen.blit(label2, (520, 400))
        screen.blit(label1, (420, 420))
        screen.blit(label2, (520, 420))
        screen.blit(label1, (420, 440))
        screen.blit(label2, (520, 440))

        if pygame.mouse.get_pressed()[0]:
            (pos1, pos2) = pygame.mouse.get_pos()
            if 420 <= pos1 <= 510 and 400 <= pos2 <= 415:
                k1 += 10
            elif 520 <= pos1 <= 610 and 400 <= pos2 <= 415:
                k1 -= 10
            elif 420 <= pos1 <= 510 and 420 <= pos2 <= 435:
                k2 += 1
            elif 520 <= pos1 <= 610 and 420 <= pos2 <= 435:
                k2 -= 1
            elif 420 <= pos1 <= 510 and 440 <= pos2 <= 455:
                k3 += 0.1
            elif 520 <= pos1 <= 610 and 440 <= pos2 <= 455:
                k3 -= 0.1

        return k1, k2, k3


class DrawLQR:
    def draw_text(self):
        myfont = pygame.font.SysFont("monospace", 15)
        label1 = myfont.render("Параметр положения тележки: %.5f" % q1, 1, (255, 0, 0))
        screen.blit(label1, (50, 400))
        label2 = myfont.render("Параметр скорости тележки: %.5f" % q2, 1, (255, 0, 0))
        screen.blit(label2, (50, 420))
        label3 = myfont.render("Параметр углового положения: %.1f" % q3, 1, (255, 0, 0))
        screen.blit(label3, (50, 440))
        label3 = myfont.render("Параметр угловой скорости: %.1f" % q4, 1, (255, 0, 0))
        screen.blit(label3, (50, 460))


    def make_buttons(q1, q2, q3, q4, lqr, pend):
        pygame.draw.rect(screen, (0, 0, 255), [420, 400, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [520, 400, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [420, 420, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [520, 420, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [420, 440, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [520, 440, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [420, 460, 90, 15])
        pygame.draw.rect(screen, (0, 0, 255), [520, 460, 90, 15])

        myfont = pygame.font.SysFont("monospace", 15)
        label1 = myfont.render("Повысить", 1, (255, 255, 255))
        label2 = myfont.render("Понизить", 1, (255, 255, 255))
        screen.blit(label1, (420, 400))
        screen.blit(label2, (520, 400))
        screen.blit(label1, (420, 420))
        screen.blit(label2, (520, 420))
        screen.blit(label1, (420, 440))
        screen.blit(label2, (520, 440))
        screen.blit(label1, (420, 460))
        screen.blit(label2, (520, 460))

        if (pygame.mouse.get_pressed()[0]):
            (pos1, pos2) = pygame.mouse.get_pos()
            if 420 <= pos1 <= 510 and 400 <= pos2 <= 415:
                q1 += 0.001
            elif 520 <= pos1 <= 610 and 400 <= pos2 <= 415:
                q1 -= 0.001
                if q1 < 0.001: q1 += 0.001
            elif 420 <= pos1 <= 510 and 420 <= pos2 <= 435:
                q2 += 0.001
            elif 520 <= pos1 <= 610 and 420 <= pos2 <= 435:
                q2 -= 0.001
            elif 420 <= pos1 <= 510 and 440 <= pos2 <= 455:
                q3 += 1000
            elif 520 <= pos1 <= 610 and 440 <= pos2 <= 455:
                q3 -= 1000
            elif 420 <= pos1 <= 510 and 460 <= pos2 <= 475:
                q4 += 10
            elif 520 <= pos1 <= 610 and 460 <= pos2 <= 475:
                q4 -= 10

        return q1, q2, q3, q4


def draw_designation():
    myfont = pygame.font.SysFont("monospace", 20)
    label1 = myfont.render("X", 1, (128, 128, 128))
    screen.blit(label1, (5, 470))
    label2 = myfont.render("PHI", 1, (128, 128, 128))
    screen.blit(label2, (0, 640))
    label3 = myfont.render("I", 1, (128, 128, 128))
    screen.blit(label3, (1785, 820))
    screen.blit(label3, (1785, 550))
    label4 = myfont.render("PI/3", 1, (128, 128, 128))
    screen.blit(label4, (25, 665))
    label5 = myfont.render("0", 1, (128, 128, 128))
    screen.blit(label5, (20, 820))
    screen.blit(label5, (20, 480))
    label6 = myfont.render("-PI/3", 1, (128, 128, 128))
    screen.blit(label6, (20, 955))
    label7 = myfont.render("550", 1, (128, 128, 128))
    screen.blit(label7, (25, 345))
    label8 = myfont.render("-550", 1, (128, 128, 128))
    screen.blit(label8, (20, 600))
    #label9 = myfont.render("100", 1, (128, 128, 128))
    #screen.blit(label9, (70, 820))
    #screen.blit(label9, (70, 480))
    label10 = myfont.render("100", 1, (128, 128, 128))
    screen.blit(label10, (135, 820))
    screen.blit(label10, (135, 480))
    #label11 = myfont.render("300", 1, (128, 128, 128))
    #screen.blit(label11, (198, 820))
    #screen.blit(label11, (198, 480))
    label12 = myfont.render("200", 1, (128, 128, 128))
    screen.blit(label12, (260, 820))
    screen.blit(label12, (260, 480))
    #label13 = myfont.render("500", 1, (128, 128, 128))
    #screen.blit(label13, (322, 820))
    #screen.blit(label13, (322, 550))
    label14 = myfont.render("300", 1, (128, 128, 128))
    screen.blit(label14, (385, 820))
    screen.blit(label14, (385, 480))
    #label15 = myfont.render("700", 1, (128, 128, 128))
    #screen.blit(label15, (447, 820))
    #screen.blit(label15, (447, 550))
    label16 = myfont.render("400", 1, (128, 128, 128))
    screen.blit(label16, (510, 820))
    screen.blit(label16, (510, 480))
    #label17 = myfont.render("900", 1, (128, 128, 128))
    #screen.blit(label17, (572, 820))
    #screen.blit(label17, (572, 550))
    label18 = myfont.render("500", 1, (128, 128, 128))
    screen.blit(label18, (635, 820))
    screen.blit(label18, (635, 480))
    #label19 = myfont.render("1100", 1, (128, 128, 128))
    #screen.blit(label19, (697, 820))
    #screen.blit(label19, (697, 550))
    label20 = myfont.render("600", 1, (128, 128, 128))
    screen.blit(label20, (760, 820))
    screen.blit(label20, (760, 480))
    #label21 = myfont.render("1300", 1, (128, 128, 128))
    #screen.blit(label21, (822, 820))
    #screen.blit(label21, (822, 550))
    label22 = myfont.render("700", 1, (128, 128, 128))
    screen.blit(label22, (885, 820))
    screen.blit(label22, (885, 480))
    #label23 = myfont.render("1500", 1, (128, 128, 128))
    #screen.blit(label23, (947, 820))
    #screen.blit(label23, (947, 550))
    label24 = myfont.render("800", 1, (128, 128, 128))
    screen.blit(label24, (1010, 820))
    screen.blit(label24, (1010, 480))
    #label25 = myfont.render("1700", 1, (128, 128, 128))
    #screen.blit(label25, (1072, 820))
    #screen.blit(label25, (1072, 550))
    label26 = myfont.render("900", 1, (128, 128, 128))
    screen.blit(label26, (1135, 820))
    screen.blit(label26, (1135, 480))
    #label27 = myfont.render("1900", 1, (128, 128, 128))
    #screen.blit(label27, (1197, 820))
    #screen.blit(label27, (1197, 550))
    label28 = myfont.render("1000", 1, (128, 128, 128))
    screen.blit(label28, (1260, 820))
    screen.blit(label28, (1260, 480))
    #label29 = myfont.render("2100", 1, (128, 128, 128))
    #screen.blit(label29, (1322, 820))
    #screen.blit(label29, (1322, 550))
    label30 = myfont.render("1100", 1, (128, 128, 128))
    screen.blit(label30, (1385, 820))
    screen.blit(label30, (1385, 480))
    #label31 = myfont.render("2300", 1, (128, 128, 128))
    #screen.blit(label31, (1447, 820))
    #screen.blit(label31, (1447, 550))
    label32 = myfont.render("1200", 1, (128, 128, 128))
    screen.blit(label32, (1510, 820))
    screen.blit(label32, (1510, 480))
    #label33 = myfont.render("2500", 1, (128, 128, 128))
    #screen.blit(label33, (1572, 820))
    #screen.blit(label33, (1572, 550))
    label34 = myfont.render("1300", 1, (128, 128, 128))
    screen.blit(label34, (1635, 820))
    screen.blit(label34, (1635, 480))
    #label35 = myfont.render("2700", 1, (128, 128, 128))
    #screen.blit(label35, (1697, 820))
    #screen.blit(label35, (1697, 550))


p = Pendulum(900, 300, pi - 30*pi / 180)
pid = PID(k1, k2, k3, p)
lqr = LQR(q1, q2, q3, q4, r, p)

while 1:
    screen.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    if event.type == pygame.KEYUP:
        if event.key == pygame.K_1:
            pid.update()
            p.move(pid.output())
            #DrawPID.draw_text(DrawPID)
            #(k1, k2, k3) = DrawPID.make_buttons(k1, k2, k3, pid, p)
        if event.key == pygame.K_2:
            lqr.update()
            p.move(lqr.output())
            #DrawLQR.draw_text(DrawLQR)
            #(q1, q2, q3, q4) = DrawLQR.make_buttons(q1, q2, q3, q4, lqr, p)
        if event.key == pygame.K_3:
            balancer = InvertedPendulumBalancer()
            balancer.run()

    pygame.event.set_blocked(pygame.MOUSEMOTION)
    pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
    pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN)
    pygame.event.set_blocked(pygame.ACTIVEEVENT)


    #pid.update()
    #lqr.update()
    #p.move(pid.output())
    #p.move(lqr.output())
    p.draw()
    #DrawPID.draw_text(DrawPID)
    #DrawLQR.draw_text(DrawLQR)
    draw_designation()
    #(k1, k2, k3) = DrawPID.make_buttons(k1, k2, k3, pid)
    #(q1, q2, q3, q4) = DrawLQR.make_buttons(q1, q2, q3, q4, lqr, p)

    clock.tick(60)
    pygame.display.flip()