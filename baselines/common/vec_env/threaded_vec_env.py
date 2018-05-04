import numpy as np
import threading
import queue

from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.subproc_vec_env import CommandHandler, CloseCommand


class WorkerThread(threading.Thread):

    CmdHandlerClass = CommandHandler

    def __init__(self, env_fn, cmd_queue, res_queue):
        """

        :param () -> gym.Env env_fn:
        :param queue.Queue cmd_queue:
        :param queue.Queue res_queue:
        """
        super().__init__(daemon=True)  # daemon so the thread is automatically killed if the main program crashes
        self.env = env_fn()
        self.cmd_handler = self.CmdHandlerClass(self.env)
        self.cmd_queue = cmd_queue
        self.res_queue = res_queue


    def run(self):
        while True:
            cmd, data = self.cmd_queue.get()
            try:
                res = self.cmd_handler.cmd(cmd, data)
                self.res_queue.put(res)
            except CloseCommand:
                break


class ThreadedVecEnv(VecEnv):

    WorkerClass = WorkerThread

    def __init__(self, env_fns, spaces=None):
        """
        :param list[gym.Env] envs: list of gym environments to run in threads
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)

        self.cmd_queues = [queue.Queue() for _ in range(nenvs)]
        self.res_queues = [queue.Queue() for _ in range(nenvs)]
        self.threads = [self.WorkerClass(env_fn, cmd_queue, res_queue)
                        for (cmd_queue, res_queue, env_fn) in zip(self.cmd_queues, self.res_queues, env_fns)]
        for t in self.threads:
            t.start()

        self.cmd_queues[0].put(('get_spaces', None))
        observation_space, action_space = self.res_queues[0].get()
        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for cmd_queue, action in zip(self.cmd_queues, actions):
            cmd_queue.put(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [res_queue.get() for res_queue in self.res_queues]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(('reset', None))
        return np.stack([res_queue.get() for res_queue in self.res_queues])

    def reset_task(self):
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(('reset_task', None))
        return np.stack([res_queue.get() for res_queue in self.res_queues])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for res_queue in self.res_queues:
                res_queue.get()
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(('close', None))
        for t in self.threads:
            t.join()
        self.closed = True

