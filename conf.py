
import math

class UUV:
    def __init__(self, x, y, z, cost, communication_cost, total_budget):
        self.x = x
        self.y = y
        self.z = z
        self.theta = 0  # Initially set to 0, can be changed if needed
        self.cost = cost
        self.communication_cost = communication_cost
        self.total_budget = total_budget
        self.CM_history = []

    def move_towards_waypoint(self, waypoint):
        dx = waypoint[0] - self.x
        dy = waypoint[1] - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            self.x += dx / distance
            self.y += dy / distance
            self.theta = math.atan2(dy, dx)

    def move_up(self):
        self.z += 1

    def move_down(self):
        self.z -= 1

    def communicate(self):
        if self.cost + self.communication_cost <= self.total_budget:
            self.cost += self.communication_cost
            return True
        return False

    def compute_CM(self, predicted_state):
        deviation = abs(self.z - predicted_state[2])
        self.CM_history.append(deviation)
        if len(self.CM_history) > 10:  # Store only the last 10 values for simplicity
            self.CM_history.pop(0)
        return deviation

    def dynamic_threshold(self):
        if not self.CM_history:
            return float('inf')  # Arbitrary high value
        avg = sum(self.CM_history) / len(self.CM_history)
        return avg * 1.2  # Threshold set to 120% of the average for simplicity


def plan_actions(uuv, waypoints):
    current_waypoint_index = 0
    actions = []

    while current_waypoint_index < len(waypoints) and uuv.cost < uuv.total_budget:
        # If UUV is close to the current waypoint, move to the next one
        if math.dist([uuv.x, uuv.y], waypoints[current_waypoint_index]) < 1.0:
            current_waypoint_index += 1

        if current_waypoint_index >= len(waypoints):
            break

        # Placeholder for the predicted state
        predicted_state = (uuv.x, uuv.y, uuv.z + 1, uuv.theta, uuv.cost)
        current_CM = uuv.compute_CM(predicted_state)
        threshold = uuv.dynamic_threshold()
        
        if current_CM > threshold:
            if uuv.communicate():
                actions.append("communicate")
            else:
                break  # If unable to communicate due to budget, stop the mission
        else:
            # Decision logic
            if uuv.z < 0:
                uuv.move_up()
                actions.append("up")
            elif uuv.z > 0:
                uuv.move_down()
                actions.append("down")
            else:
                uuv.move_towards_waypoint(waypoints[current_waypoint_index])
                actions.append("move_towards_waypoint")

    return actions

# Example usage
uuv = UUV(0, 0, -5, 0, 10, 100)  # Starting 5 units below sea level
waypoints = [(100, 10), (20, 5), (30, 15)]
actions = plan_actions(uuv, waypoints)
print(actions)
