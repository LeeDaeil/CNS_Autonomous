import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        """
        ReplayBuffer를 선언 해줌.
        :param max_size: int
        :param input_shape: 튜플 형태로 입력 받음. ex. (1, 3)
        :param n_actions: int 형식
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))         # Shape: (mem_size, *input_shape)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))     # Shape: (mem_size, *input_shape)
        self.action_memory = np.zeros((self.mem_size, n_actions))           # Shape: (mem_size, n_action)
        self.reward_memory = np.zeros(self.mem_size)                        # Shape: (mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)       # Shape: (mem_size)

    def store_transition(self, state, action, reward, state_, done):
        # 현재 count 를 최대 mem_size 로 나누어 한정된 메모리 주소에서 순환하도록 설계함.
        index = self.mem_cntr % self.mem_size

        # 데이터를 저장
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # 1] 앞에서 메모리가 초기화 된 상태이기 때문에, 초기화된 값을 가져오지 않도록 현재 카운터보다 작은 범위에서 순환
        max_mem = min(self.mem_cntr, self.mem_size)     # int

        # 2] 0 ~ max_mem 범위 내에서 batch_size 의 갯수 만큼 index 값을 생성함.
        batch = np.random.choice(max_mem, batch_size)

        # 3] 랜덤으로 생성된 index 에 따라서 샘플링
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


if __name__ == '__main__':
    a = ReplayBuffer(max_size=3, input_shape=(3, 1), n_actions=4)
    a.store_transition(1, 1, 1, 1, 1)
    a.store_transition(1, 1, 1, 1, 1)
    a.store_transition(1, 1, 1, 1, 1)
    a.store_transition(1, 1, 1, 1, 1)
    a.store_transition(1, 1, 1, 1, 1)
    a.store_transition(1, 1, 1, 1, 1)