import pygame
from pygame import gfxdraw
from rl.gaze_capture.face_processor import FaceProcessor
from rl.gaze_capture.ITrackerModel import ITrackerModel
import cv2
import torch
import numpy as np
import math
import h5py


predictor_path = './image/rl/gaze_capture/model_files/shape_predictor_68_face_landmarks.dat'
webcam = cv2.VideoCapture(0)
face_processor = FaceProcessor(predictor_path)

i_tracker = ITrackerModel()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    i_tracker.cuda()
    state = torch.load('image/rl/gaze_capture/checkpoint.pth.tar')['state_dict']
else:
    device = "cpu"
    state = torch.load('image/rl/gaze_capture/checkpoint.pth.tar',
                       map_location=torch.device('cpu'))['state_dict']
i_tracker.load_state_dict(state, strict=False)

pygame.init()
screen = pygame.display.set_mode()
width, height = screen.get_size()
text_field_height = height / 30
header_coord = (width / 2, text_field_height)
fonts = {font_size: pygame.font.Font('freesansbold.ttf', font_size) for font_size in range(2, 34, 2)}


def draw_rect_with_text(text, text_color, width, height, rect_color=None, font_size=32, center=None,
                        left=0, top=0):
    while font_size > 2:
        font = fonts[font_size]
        req_width, req_height = font.size(text)
        if req_width <= width and req_height <= height:
            break
        font_size -= 2
    rect = pygame.Rect(left, top, width, height)
    if center is not None:
        rect.center = center
    if rect_color is None:
        rect_color = (0, 0, 0)
    pygame.draw.rect(screen, rect_color, rect)
    text_img = font.render(text, True, text_color)
    if center is not None:
        text_rect = text_img.get_rect(center=center)
    else:
        text_rect = text_img.get_rect(left=left, top=top)
    screen.blit(text_img, text_rect)
    return text_rect


def draw_circle_with_text(text, text_color, center, radius, circle_color=None, font_size=32):
    font = fonts[font_size]
    while font_size > 2:
        req_width, req_height = font.size(text)
        if req_width <= 2 * radius and req_height <= 2 * radius:
            break
        font_size -= 2
        font = fonts[font_size]
    center = np.round(center).astype(int)
    if circle_color is None:
        circle_color = (0, 0, 0)
    gfxdraw.aacircle(screen, center[0], center[1], radius, circle_color)
    gfxdraw.filled_circle(screen, center[0], center[1], radius, circle_color)
    text_img = font.render(text, True, text_color)
    text_rect = text_img.get_rect(center=center)
    screen.blit(text_img, text_rect)


calibration_points = [(width / 2 + 350, height / 2), (width / 2, height / 2), (width / 2 - 350, height / 2)]
screen.fill((0, 0, 0))
for i, point in enumerate(calibration_points):
    color = (255, 255, 255)
    draw_circle_with_text(str(i), (0, 0, 0), point, 20, (255, 255, 255))

draw_rect_with_text('Press SPACE to start calibration', (255, 255, 255), width, text_field_height, center=header_coord)
pygame.display.flip()

done = False
running = True
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                done = True
                break

draw_rect_with_text("Look at the highlighted number", (255, 255, 255), width,
                    text_field_height, center=header_coord)

cycles = 0
n_points = len(calibration_points)
remaining = [50] * n_points
curr_point = None


def get_event():
    global running
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False


features = [[] for i in range(len(calibration_points))]
count = 0
while running and sum(remaining) > 0:
    if curr_point is not None:
        draw_circle_with_text(str(curr_point), (0, 0, 0), calibration_points[curr_point],
                              20, (255, 255, 255))

    curr_point = np.random.choice(np.nonzero(remaining)[0])
    curr_coord = calibration_points[curr_point]
    draw_circle_with_text(str(curr_point), (0, 0, 0), curr_coord, 20, (255, 165, 0))

    pygame.display.flip()
    get_event()

    pygame.time.wait(1000)
    gaze_features = None
    while gaze_features is None:
        _, frame = webcam.read()
        gaze_features = face_processor.get_gaze_features(frame)
    features[curr_point].append(gaze_features)
    remaining[curr_point] = remaining[curr_point] - 1
    count += 1
    print(count)

dataset = h5py.File('image/rl/gaze_capture/gaze_data_eval.h5')
for i in range(len(features)):
    data = []
    if len(features[i]) > 0:
        point = zip(*features[i])
        point = [torch.from_numpy(np.array(feature)).float().to(device) for feature in point]

        batch_size = 32
        n_batches = math.ceil(len(point[0]) / batch_size)
        for j in range(n_batches):
            batch = [feature[j * batch_size: (j + 1) * batch_size] for feature in point]
            output = i_tracker(*batch)
            data.extend(output.detach().cpu().numpy())
        dataset.create_dataset(str(i), data=data)

dataset.close()
