import numpy as np

from collections import deque

from environment import GridWorld


# =========================
# 1. Replay Buffer
# =========================

class ReplayBuffer:

    def __init__(
        self,
        capacity
    ):

        self.buffer = deque(
            maxlen=capacity
        )


    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        transition = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.buffer.append(
            transition
        )


    def sample(
        self,
        batch_size,
        rng
    ):

        indices = rng.choice(
            len(self.buffer),
            size=batch_size,
            replace=False
        )

        batch = [
            self.buffer[i]
            for i in indices
        ]

        return batch


    def __len__(self):

        return len(
            self.buffer
        )


# =========================
# 2. Environment
# =========================

env = GridWorld()

rng = np.random.default_rng(
    42
)


# =========================
# 3. Create Replay Buffer
# =========================

buffer = ReplayBuffer(
    capacity=20
)


# =========================
# 4. Collect Experience
# =========================

state = env.reset()


for step in range(15):

    action = rng.integers(4)

    next_state, reward, done = env.step(
        action
    )


    buffer.push(
        state,
        action,
        reward,
        next_state,
        done
    )


    state = next_state


    if done:

        state = env.reset()


# =========================
# 5. Buffer Info
# =========================

print(
    "Buffer size:",
    len(buffer)
)


print(
    "\nAll transitions:"
)

for i, transition in enumerate(
    buffer.buffer
):

    print(
        i,
        transition
    )


# =========================
# 6. Random Sample
# =========================

batch_size = 5

batch = buffer.sample(
    batch_size,
    rng
)


print(
    "\nRandom Batch:"
)

for transition in batch:

    print(
        transition
    )