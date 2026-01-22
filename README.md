# bug description

When running animal ai if the agent hits a Goal object that ends the episode when reached, this resets the environment. However, in order to call `step()` again, the python code must first call `reset()` as reset must be called after each step where `done=True` is returned. This means that the environment is reset twice in a row, which will skip the following arena.

# steps to reproduce

1. set up an animal ai installation
2. run the following command:

```bash
python generate_trajectories.py --config_path coloured_wall_arena.yml --AAI_path <path to animal ai binary>
```
3. you should observe that the the first arena has a large red wall in front of the agent. The agent will move around to the left side of the wall and reach the goal, which ends the episode. However, when the next arena is loaded, the large red wall is still there instead of the second arena which uses a blue wall. The agent will move around the right side of the red wall and fail to reach the goal. 
4. Once the episode ends due to something other than reaching the goal, the next arena will load correctly with a blue wall showing we are in the second arena. and the process will repeat alternating arenas because the agent will not reach the goal in either arena.

# potential solution 

as a quick fix I added a check in the python code to see if the `done=True` was the reason the episode ended, and if so, it calls reset a number of times equal to the number of arenas -1 to get to the next arena. this fix can be toggled by changing the variable `ENABLE_FIX` on line 18 of generate_trajectories.py.