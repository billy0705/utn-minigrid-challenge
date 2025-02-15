from openai import OpenAI
from typing import Dict, Any, Tuple, Optional
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX
from collections import deque

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

ACTION_MAP = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}


def relative_to_absolute(agent_direction, relative_direction):
    if agent_direction == "north":
        if relative_direction == "left":
            return "west"
        elif relative_direction == "right":
            return "east"
        elif relative_direction == "front":
            return "north"
    elif agent_direction == "south":
        if relative_direction == "left":
            return "east"
        elif relative_direction == "right":
            return "west"
        elif relative_direction == "front":
            return "south"
    elif agent_direction == "east":
        if relative_direction == "left":
            return "north"
        elif relative_direction == "right":
            return "south"
        elif relative_direction == "front":
            return "east"
    elif agent_direction == "west":
        if relative_direction == "left":
            return "south"
        elif relative_direction == "right":
            return "north"
        elif relative_direction == "front":
            return "west"
    else:
        raise ValueError(f"Invalid agent direction: {agent_direction}")


class Agent:
    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", api_url: Optional[str] = None
    ):
        """
        Initialize the agent.

        Args:
            api_key: API key
            model: model to use
            temperature: Temperature for model sampling
        """
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model = model
        self.temperature = 0.5
        self.past_states = deque(maxlen=2)  # [state, response]
        self.current_step = 0
        self.coordinate = (0, 0)
        self.objects_coordinates = {}

        # System prompt to explain the task

    def find_last_action(self, action_text, action_map):
        action_idx = None
        last_position = -1
        found_action = None

        # Check each possible action
        for idx, text in action_map.items():
            # Find the last position of this action in the text
            position = action_text.rfind(text)

            # If found and it's later than our previous match
            if position != -1 and position > last_position:
                last_position = position
                action_idx = idx
                found_action = text

        return action_idx, found_action

    def get_system_prompt(self, direction, mission):
        objs = ""
        if len(self.objects_coordinates) > 0:
            objs = "\n".join([f" * {v} is at {k}" for k, v in self.objects_coordinates.items()])
        forward_coordate = (self.coordinate[0] + (1 if direction == "east" else -1 if direction == "west" else 0),
                            self.coordinate[1] + (-1 if direction == "south" else 1 if direction == "north" else 0))
        return f"""You are an agent in a grid-world environment. The goal is to navigate the world and interact with objects to complete the mission.
Mission: 
{mission}
Objects found in the environment:
{objs}
        
Available Actions:
1. MOVEMENT:
   - turn left: Rotate 90° counterclockwise to face {relative_to_absolute(direction, 'left')}
   - turn right: Rotate 90° clockwise to face {relative_to_absolute(direction, 'right')}
   - move forward: Advance one cell in direction {direction}, if you choose this action the next coordinate is {forward_coordate}

2. OBJECT INTERACTIONS:
   - pick up: Collect an object from 1 cell in front of agent
   - drop: Release currently held object into the cell directly in front of agent
   - toggle: Interact with doors or boxes 1 cell in front of agent

Environmental Rules:
- Navigation:
  * You can face four directions: north, south, east, west
  * x-asix is horizontal (east or west) and y-axis is vertical (north or south)
  * x-axis positive is east, y-axis positive is north
  * y-axis negative is south, x-axis negative is west
  * Don't do the object interaction if it is not 1 cell in front of you or there is no object
  * Objects are solid and must be navigated around
  * Each action moves exactly one cell or rotates 90 degrees
  * Don't move forward or turn to the walls or closed doors
  
  
- Object Interaction Rules:
  Important: **Every Interaction must be directly in front of an object (not 1 cell left or right) to interact with it**
  * Keys:
    - Can be picked up when it is only 1 cell in front of you
    - Must be in your inventory to unlock doors
    - Only one key can be carried at a time
  * Doors:
    - Must have matching key to toggle, only toggle it when door is 1 cell in front of you
    - Must be directly in front of you to interact
  * Boxes:
    - Must be 1 cell in front of you to toggle, must facing to the box
    - May contain keys or other objects
    - Contents are only revealed upon opening
    - You can only pick up boxes if you have no object in your hand

Planning Guidelines:
1. If target not visible:
   - Implement systematic exploration
   - Look for keys that might be needed
2. If target visible but unreachable:
   - Plan optimal path accounting for obstacles
   - Consider if keys are needed for access
3. For locked doors:
   - Prioritize to search for keys before attempting to unlock doors
   - Search for keys in boxes and open areas
   - Once open door, don't close it
4. Same Observation:
    - Don't repeat the same action

Think about how to solve the mission?  Think about the priority things to do. Give me what you are thinking and why you are taking this action. At the end give me the next step action. After this action, don't give me anything else."""

    def parse_observation(self, obs: Dict[str, Any], mission: str) -> str:
        """
        Convert the observation into a text prompt for the model.

        Args:
            obs: Observation from the environment
            mission: Current mission string

        Returns:
            Formatted prompt string
        """
        # Convert direction number to cardinal direction
        directions = ["east", "south", "west", "north"]
        direction = directions[obs["direction"]]

        # Parse the grid to find visible objects
        visible_objects = []
        grid = obs["image"]
        # print(f"{grid[:,:,0]=}")
        # print(f"{self.coordinate=}")
        infront = ""
        if grid[3, 5, 0] == 2:
            infront = "wall"
        elif grid[3, 5, 0] > 2:
            infront = f"{IDX_TO_COLOR[grid[3, 5, 1]]} {IDX_TO_OBJECT[grid[3, 5, 0]]}"
        elif grid[3, 5, 0] == 1:
            infront = "empty cell"

        # Convert object types to descriptions
        for x in range(7):
            for y in range(7):
                if x == 3 and y == 6:
                    continue  # skip for agent position - it's the object being held
                obj_id, color_id, door_state = grid[x, y]
                if obj_id > 2:
                    obj_state = ""
                    if obj_id == 4:  # it's a door
                        obj_state = f"{IDX_TO_STATE[door_state]} "
                    obj_name = f"{obj_state}{IDX_TO_COLOR[color_id]} {IDX_TO_OBJECT[obj_id]}"
                    obj_repr = f"\n * {obj_name} -"
                    obj_pos = ""
                    distance = abs(x - 3) + abs(y - 6)
                    if x < 3:
                        obj_pos += f" {3 - x} cells to the left"
                    elif x > 3:
                        obj_pos += f" {x - 3} cells to the right"
                    if y < 6:
                        if obj_pos != "":
                            obj_pos += " AND"
                        obj_pos += f" {6 - y} cells in the front"
                    obj_repr = obj_repr + obj_pos + f" ({distance} cells away)"
                    visible_objects.append(obj_repr)
                    if direction == "west":
                        obj_x = self.coordinate[0] - (6 - y)
                        obj_y = self.coordinate[1] - (3 - x)
                        self.objects_coordinates[(obj_x, obj_y)] = obj_name
                    elif direction == "south":
                        obj_x = self.coordinate[0] + (3 - x)
                        obj_y = self.coordinate[1] - (6 - y)
                        self.objects_coordinates[(obj_x, obj_y)] = obj_name
                    elif direction == "east":
                        obj_x = self.coordinate[0] + (6 - y)
                        obj_y = self.coordinate[1] + (3 - x)
                        self.objects_coordinates[(obj_x, obj_y)] = obj_name
                    elif direction == "north":
                        obj_x = self.coordinate[0] - (3 - x)
                        obj_y = self.coordinate[1] + (6 - y)
                        self.objects_coordinates[(obj_x, obj_y)] = obj_name

        actionable_object = "none"
        if grid[3, 5, 0] > 2:
            actionable_object = (
                f"{IDX_TO_COLOR[grid[3, 5, 1]]} {IDX_TO_OBJECT[grid[3, 5, 0]]}"
            )
        holding_object = "none"
        if grid[3, 6, 0] > 2:
            holding_object = (
                f"{IDX_TO_COLOR[grid[3, 6, 1]]} {IDX_TO_OBJECT[grid[3, 6, 0]]}"
            )

        walls = []
        if grid[2, 6, 0] == 2:
            walls.append(f"left ({relative_to_absolute(direction, 'left')})")
        if grid[4, 6, 0] == 2:
            walls.append(f"right ({relative_to_absolute(direction, 'right')})")
        if grid[3, 5, 0] == 2:
            walls.append(f"front ({relative_to_absolute(direction, 'front')})")
        if len(walls) == 0:
            walls.append("none")

        Wall_info = "No wall around"
        if grid[2, 6, 0] == 2:
            if Wall_info == "No wall around":
                Wall_info = "Wall on the left"
        elif grid[4, 6, 0] == 2:
            if Wall_info == "No wall around":
                Wall_info = "Wall on the right"
            else:
                Wall_info += f", Wall on the right"
        elif grid[3, 5, 0] == 2:
            if Wall_info == "No wall around":
                Wall_info = f"Wall in front"
            else:
                Wall_info += f", Wall in front"

        # Create the prompt
        past_states_str = "\n".join(self.past_states)
        current_state = f"""[Step {self.current_step}]
- Agent position: {self.coordinate}
- Facing '{direction}'
- Wall infomation: {Wall_info}
- Cell in front: {infront}
- Visible objects: {', '.join(visible_objects) if visible_objects else 'none'}
- Actionable object: {actionable_object}
- Holding object: {holding_object}"""
        prompt = f"""Current observation:
        {current_state}
        Previous observations:
{past_states_str}
"""

        return prompt, current_state, direction

    def get_action(self, obs: Dict[str, Any], mission: str, verbose: bool) -> int:
        """
        Get the next action from the model.

        Args:
            obs: Observation from the environment
            mission: Current mission string

        Returns:
            Action index
        """
        prompt, current_state, direction = self.parse_observation(obs, mission)
        final_prompt = f"{self.get_system_prompt(direction, mission)}\n\n{prompt}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": final_prompt},
            ],
            temperature=self.temperature,
            max_tokens=1000,
        )
        if verbose:
            print("==================================")
            print("final_prompt:\n", final_prompt)
            print("response:\n", response.choices[0].message.content)

        response = response.choices[0].message.content.strip().lower()

        action_idx, action_text = self.find_last_action(response, ACTION_MAP)

        if action_idx is None:
            print(
                f"Warning: Invalid action '{action_text}', defaulting to move forward"
            )
            action_idx = 2  # Default to move forward
            action_text = ACTION_MAP[2]

        self.past_states += [
            current_state,
            f"Response: {action_text}",
        ]
        self.current_step += 1

        # dict with metadata to log during eval
        metadata = {
            "final_prompt": final_prompt,
            "response": response,
            "action_text": action_text,
        }
        if action_idx == 2:
            if obs["image"][3, 5, 0] < 2:
                self.coordinate = (
                    self.coordinate[0] + (1 if direction == "east" else -1 if direction == "west" else 0),
                    self.coordinate[1] + (-1 if direction == "south" else 1 if direction == "north" else 0),
                )

        return action_idx, metadata


def handle_state(
    obs: Dict[str, Any], mission: str, agent: Agent, verbose: bool = False
) -> int:
    """
    Process the current state and get the next action.

    Args:
        obs: Current observation from the environment
        mission: Current mission string
        agent: Agent instance
        verbose: Whether to print debug information

    Returns:
        Action index to take
    """

    action, metadata = agent.get_action(obs, mission, verbose)

    if verbose:
        print("Chosen Action:", ACTION_MAP[action])

    return action, metadata
