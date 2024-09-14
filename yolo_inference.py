from ultralytics import YOLO

model = YOLO('models/tennis_ball.pt')

result = model.track('input_videos\input_video.mp4', conf = 0.2, save= True)
# print(result)
# print('Boxes:')
# for box in result[0].boxes:
#     print(box)