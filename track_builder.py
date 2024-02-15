import pygame
from Track import Track


# Example file showing a circle moving on screen
WIDTH, HEIGHT = 1280, 720
SCREEN_CENTER = (WIDTH / 2, HEIGHT / 2)


# pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
dt = 0

track_surface = pygame.Surface((WIDTH, HEIGHT))
track = Track(track_surface, loop=True)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            point = pygame.mouse.get_pos()
            track.add_point(point)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                track.remove_point()
            if event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                track.save()

    # fill the screen with a color to wipe away anything from last frame
    track_surface.fill("black")
    screen.fill("white")
    if track.checkpoints:
        click = pygame.mouse.get_pressed()[0]
        if click:
            track.get_side_points()

    track.draw_poly()
    track.draw()

    screen.blit(track_surface, (0, 0))

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(30) / 1000

pygame.quit()
