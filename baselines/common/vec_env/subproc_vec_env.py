import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


class CloseCommand(Exception):
    pass


class CommandHandler:
    def __init__(self, env):
        """

        :param gym.Env env:
        """
        self.env = env

    def cmd(self, cmd, data):
        env = self.env
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            return (ob, reward, done, info)
        elif cmd == 'reset':
            ob = env.reset()
            return ob
        elif cmd == 'reset_task':
            ob = env.reset_task()
            return ob
        elif cmd == 'close':
            raise CloseCommand
        elif cmd == 'get_spaces':
            return (env.observation_space, env.action_space)
        else:
            raise NotImplementedError('unknown command "%s" with data %s' % (cmd, data))


class WorkerProcess(Process):

    CmdHandlerClass = CommandHandler

    def __init__(self, remote, parent_remote, env_fn_wrapper, group=None, name=None):
        """

        :param Pipe remote:
        :param Pipe parent_remote:
        :param env_fn_wrapper:
        :param group:
        :param name:
        """
        super().__init__(group=group, name=name)
        self.env = env_fn_wrapper.x()
        self.remote = remote
        self.parent_remote = parent_remote
        self.cmd_handler = self.CmdHandlerClass(self.env)

    def run(self):
        self.parent_remote.close()
        while True:
            cmd, data = self.remote.recv()
            try:
                res = self.cmd_handler.cmd(cmd, data)
                self.remote.send(res)
            except CloseCommand:
                self.remote.close()
                break


class SubprocVecEnv(VecEnv):

    WorkerClass = WorkerProcess

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [self.WorkerClass(work_remote, remote, CloudpickleWrapper(env_fn))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
