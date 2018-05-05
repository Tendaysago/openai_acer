
from baselines.common.vec_env.threaded_vec_env import CommandHandler, WorkerThread, ThreadedVecEnv


class RogueCommandHandler(CommandHandler):

    def cmd(self, cmd, data):
        if cmd == 'stats':
            return self.env.unwrapped.stats()
        elif cmd == 'save_state':
            checkpoint_dir, id = data
            return self.env.unwrapped.save_state(checkpoint_dir, id)
        elif cmd == 'restore_state':
            checkpoint_dir, id = data
            return self.env.unwrapped.restore_state(checkpoint_dir, id)
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

    def save_state(self, checkpoint_dir, global_t):
        for i, cmd_queue in enumerate(self.cmd_queues):
            cmd_queue.put(('save_state', (checkpoint_dir, '%s-%s' % (i, global_t))))
        for res_queue in self.res_queues:
            res_queue.get()

    def restore_state(self, checkpoint_dir, global_t):
        for i, cmd_queue in enumerate(self.cmd_queues):
            cmd_queue.put(('restore_state', (checkpoint_dir, '%s-%s' % (i, global_t))))
        for res_queue in self.res_queues:
            res_queue.get()
