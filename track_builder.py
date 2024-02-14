import pygame


# Example file showing a circle moving on screen
WIDTH, HEIGHT = 1280, 720
SCREEN_CENTER = (WIDTH / 2, HEIGHT / 2)


# pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
dt = 0

checkpoints = []


def get_side_points(point: tuple[int, int]):
    pos = pygame.mouse.get_pos()
    dx, dy = pos[0] - point[0], pos[1] - point[1]
    opp_pos = point[0] - dx, point[1] - dy
    return pos, opp_pos


while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            print(event.type)
            point = pygame.mouse.get_pos()
            checkpoints.append([point, None, None])
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                if checkpoints:
                    checkpoints.pop()

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("purple")
    if checkpoints:
        click = pygame.mouse.get_pressed()[0]
        if click:
            pos, pos_opp = get_side_points(checkpoints[-1][0])
            checkpoints[-1][1] = pos
            checkpoints[-1][2] = pos_opp
        for checkpoint in checkpoints:
            pygame.draw.circle(screen, "green", checkpoint[0], 5)
            pygame.draw.circle(screen, "red", checkpoint[1], 5)
            pygame.draw.circle(screen, "yellow", checkpoint[2], 5)

    inner_lines = [ch[2] for ch in checkpoints]
    outer_lines = [ch[1] for ch in checkpoints]
    if len(checkpoints) < 2:
        pass
    else:
        pygame.draw.lines(screen, "red", False, outer_lines)
        pygame.draw.lines(screen, "yellow", False, inner_lines)
        pygame.draw.line(screen, "yellow", inner_lines[-1], inner_lines[0])
        pygame.draw.line(screen, "red", outer_lines[-1], outer_lines[0])

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(30) / 1000

pygame.quit()
