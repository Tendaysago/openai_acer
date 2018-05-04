
from baselines.common.vec_env.threaded_vec_env import CommandHandler, WorkerThread, ThreadedVecEnv


class RogueCommandHandler(CommandHandler):

    def cmd(self, cmd, data):
        if cmd == 'stats':
            return self.env.unwrapped.stats()
        else:
            return super().cmd(cmd, data)


class RogueWorkerThread(WorkerThread):

    CmdHandlerClass = RogueCommandHandler


class RogueThreadedVecEnv(ThreadedVecEnv):

    WorkerClass = RogueWorkerThread

    def stats(self):
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(('stats', None))
        return [res_queue.get() for res_queue in self.res_queues]
