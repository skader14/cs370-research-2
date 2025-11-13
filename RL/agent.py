import sys
import json

print("Python agent ready", flush=True)

while True:
    line = sys.stdin.readline().strip()
    if not line:
        continue

    if line == "":
        continue

    try:
        state = json.loads(line)
    except json.JSONDecodeError:
        print(json.dumps({"error": "invalid json"}), flush=True)
        continue

    # Shutdown command from Java
    if "shutdown" in state:
        print(json.dumps({"ack": "shutdown"}), flush=True)
        break

    # TEST ACTION:
    # Just echo the state back with a tag
    action = {
        "ack": "received",
        "original_state": state
    }

    print(json.dumps(action), flush=True)
