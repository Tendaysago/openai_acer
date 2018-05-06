
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, WorkerProcess
from .threaded_vec_env import RogueCommandHandler


class RogueWorkerProcess(WorkerProcess):

    CmdHandlerClass = RogueCommandHandler


class RogueSubprocVecEnv(SubprocVecEnv):

    WorkerClass = RogueWorkerProcess

    def stats(self):
        for remote in self.remotes:
            remote.send(('stats', None))
        return [remote.recv() for remote in self.remotes]

    def save_state(self, checkpoint_dir, global_t):
        for i, remote in enumerate(self.remotes):
            remote.send(('save_state', (checkpoint_dir, '%s-%s' % (i, global_t))))
        for remote in self.remotes:
            remote.recv()

    def restore_state(self, checkpoint_dir, global_t):
        for i, remote in enumerate(self.remotes):
            remote.send(('restore_state', (checkpoint_dir, '%s-%s' % (i, global_t))))
        for remote in self.remotes:
            remote.recv()
