import os
import vizdoom as vzd

game = vzd.DoomGame()  # type: ignore
game.load_config(os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"))
game.init()
for _ in range(1000):
    state = game.get_state()
    action = game.action_space.sample()
    reward = game.make_action(action)

    if game.is_episode_finished():
        game.new_episode()

game.close()
