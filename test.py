import cozmo
import time
from iudrl_agent import IUDRL_Agent as Agent
from torchvision.transforms.functional import *

straight_drive = lambda robot: robot.drive_straight(cozmo.util.Distance(100), cozmo.util.Speed(100))
straight_drive_backwards = lambda robot: robot.drive_straight(cozmo.util.Distance(-100), cozmo.util.Speed(100))
rot_90_left = lambda robot: robot.turn_in_place(cozmo.util.Angle(degrees=30))
rot_90_right = lambda robot: robot.turn_in_place(cozmo.util.Angle(degrees=-30))
do_nothing = lambda robot: robot.drive_straight(cozmo.util.Distance(0), cozmo.util.Speed(0))
move_head_up = lambda robot: robot.set_head_angle(cozmo.util.Angle(degrees=robot.head_angle.degrees+5))
move_head_down = lambda robot: robot.set_head_angle(cozmo.util.Angle(degrees=robot.head_angle.degrees-5))

def get_shot(robot: cozmo.robot.Robot, size):
    image = robot.world.latest_image.raw_image
    pillow = resize(image, size)
    tensor = to_tensor(pillow)
    tensor = torch.unsqueeze(tensor, dim=0)
    return tensor

def main(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    time.sleep(1)
    agent = Agent((100,100), 7)
    state = get_shot(robot, (100,100))
    replay_process = torch.multiprocessing.Process(target=agent.replay)
    acc = 0
    while True:
        action, probs, features = agent.act(state)
        try:
            if action == 0:
                straight_drive(robot)
            elif action == 1:
                straight_drive_backwards(robot)
            elif action == 2:
                rot_90_left(robot)
            elif action == 3:
            	rot_90_right(robot)

            next_state = get_shot(robot, (100,100))
            reward = 0
            agent.remember(features, state, probs, reward, next_state, acc)
            if acc == 0:
                replay_process.start()

            prev_state = state
            state = next_state
        except cozmo.exceptions.RobotBusy:
            pass

        acc += 1

cozmo.run_program(main)