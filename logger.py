import torch


class Logger(object):
    def __init__(self, runs, sv, strr, ste, info=None):
        self.info = info
        self.sv = sv
        self.strr = strr
        self.ste = ste
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print("Highest Train: {} +/- {}".format(r.mean(), r.std()))

            r = best_result[:, 1]
            print("Highest Valid: {} +/- {}".format(r.mean(), r.std()))
            self.sv.append(r.mean())


            r = best_result[:, 2]
            print("Final Train: {} +/- {}".format(r.mean(), r.std()))
            self.strr.append(r.mean())


            r = best_result[:, 3]
            print("Final Test: {} +/- {}".format(r.mean(), r.std()))
            self.ste.append(r.mean())






        
