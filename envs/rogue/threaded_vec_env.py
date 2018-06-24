
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

    @staticmethod
    def aggregate_stats(stats, list_keys=('lvls_avg', 'alvls_avg')):
        """
        Aggregates list stats of different envs by padding with zeros,
        such that all lists for that key have the same length

        :param list[dict] stats:
            stats to aggregate
        :param iterable[str] list_keys:
            keys corresponding to a list stat

        :return:
            aggregated stats
        """
        for key in list_keys:
            max_len = 0
            for s in stats:
                try:
                    s_len = len(s[key])
                except KeyError:
                    s_len = 0
                max_len = max(max_len, s_len)
            if max_len > 0:
                for s in stats:
                    try:
                        list_avg = s[key]
                    except KeyError:
                        list_avg = []
                        s[key] = list_avg
                    diff = max_len - len(list_avg)
                    if diff > 0:
                        list_avg.extend([0]*diff)
        return stats

    def stats(self):
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(('stats', None))
        return self.aggregate_stats([res_queue.get() for res_queue in self.res_queues])

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
