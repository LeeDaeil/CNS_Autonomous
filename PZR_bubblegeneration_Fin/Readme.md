# SAC-Discrete Controller for Cold Shutdown Operation
## 1. Purpose
    - Cold Shutdown 부터 Hot Shutdown까지 가압기 압력 및 수위 제어 컨트롤러를 개발하는 것이 목표
## 2. Method
    [1] RL algorithm으로 SAC-Discrete 적용
    [2] 분산 학습과 학습 메모리의 효율성을 높이기 위해서 Distributed prioritized experience replay 방법 적용
## 3. Update History
    [2021-01-02] Code 빌드 시작
## 4. Result
    [-] SAC-Discrete 에이전트와 PID 제어기와의 성능 비교 수행
## 5. Reference
    [1] Christodoulou, Petros. "Soft actor-critic for discrete action settings." arXiv preprint arXiv:1910.07207 (2019).
    [2] Horgan, Dan, et al. "Distributed prioritized experience replay." arXiv preprint arXiv:1803.00933 (2018).
    [3] 
