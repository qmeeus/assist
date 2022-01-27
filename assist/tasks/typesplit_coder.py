import numpy as np
import random

from assist.tasks import Task, coder
from assist.tools import logger


class TypeSplitCoder(coder.Coder):
    ''' a Coder that does not shares the places for args with the same type'''

    def __init__(self, structure, conf):
        super(TypeSplitCoder, self).__init__(structure, conf)

        if self.conf['tasklabel'] == 'True':
            self.taskindices = {t: i for i, t in enumerate(structure.tasks)}
            index = len(structure.tasks)
        else:
            index = 0

        #give an index to each task and its arguments
        self.argindices = dict()
        for task in structure.tasks:
            self.argindices[task] = dict()

            if not structure.tasks[task] and self.conf['tasklabel'] != 'True':
                self.argindices[task]['task'] = dict()
                self.argindices[task]['task'][task] = index
                index += 1

            for arg in structure.tasks[task]:
                argtype = structure.types[structure.tasks[task][arg]]
                if type(argtype).__name__ == 'Enumerable':

                    #give an index to each option in the enumerable
                    self.argindices[task][arg] = dict()

                    for option in argtype.options:
                        self.argindices[task][arg][option] = index
                        index += 1

        #save the number of labels
        self.numlabels = index

    def encode(self, task, noisetype=None, noiseprob=None):
        '''
        Encode the task representation into a 1D vector
        Parameters
        ----------
        task : Task
            An object representing a task (target intent)
        noisetype: str
            'None': noise-free encoding; noiseprob ignored
            'Deletion': one of the 1-s in the encoded vector replaced with 0 w.p. noiseprob
            'Insertion': one of the 0-s in the encoded vector replaced with 1 w.p. noiseprob
            'Value': EVERY slot value is replaced with another valid one (same task) w.p. noiseprob.
        noiseprob: float
            Probabilty of replacing values in the encoded vector (between 0 and 1)
        Returns:
            the encoded task representation as a numpy array
        '''

        noisetype = noisetype or self.conf["noisetype"]
        noiseprob = noiseprob or self.conf["noiseprob"]

        vector = np.zeros([self.numlabels])

        #check the correctness of the task representation
        if task.name not in self.argindices:
            raise Exception('unknown task %s' % task.name)

        if self.conf['tasklabel'] == 'True':
            vector[self.taskindices[task.name]] = 1

        if not task.args and self.conf['tasklabel'] != 'True':
            vector[self.argindices[task.name]['task'][task.name]] = 1

        #put the argument indices to one
        for arg in task.args:
            if arg not in self.argindices[task.name]:
                raise Exception('unknown argument %s for task %s' %
                                (arg, task.name))

            option = task.args[arg]

            if option != '':

                if option not in self.argindices[task.name][arg]:
                    raise Exception('unknown %s for argument %s in task %s'
                                    % (option, arg, task.name))
                if noisetype == 'Value' and random.random() < noiseprob:
                    # modify to a random sibling
                    key = '%s.%s' % (task.name,arg)
                    siblings = self.slotids[key]
                    if len(siblings) > 1:
                        siblings.remove(self.argindices[task.name][arg][option])
                        vector[random.choice(siblings)] = 1
                else:
                    vector[self.argindices[task.name][arg][option]] = 1

        if noisetype == 'Deletion':
            #select a 1 from vector
            victim = random.choice(np.nonzero(vector)[0])
            if random.random() < noiseprob:
                vector[victim] = 0

        elif noisetype == 'Insertion':
            #select a 0 from vector
            victim = random.choice(np.argwhere(vector == 0))
            if random.random() < noiseprob:
                vector[victim] = 1

        return vector

    def decode(self, vector, cost):
        '''get the most likely task representation for the vector

        Args:
            vector: the vector to decode
            cost: a callable: cost(hypothesis, vector) that returns a cost for
                a hypothesis
        Returns:
            a task representation'''

        #threshold the vector
        threshold = float(self.conf['threshold'])
        if threshold > 0:
            vector = np.where(vector > threshold, vector, 0)

        if (vector == 0).all():
            logger.error("Probability distribution is null")

        best_candidate, best_score = (None, 0)
        for task in self.argindices:

            args = {}
            for arg in self.argindices[task]:
                if arg == 'task':
                    continue
                argvec = vector[list(self.argindices[task][arg].values())]
                if not np.any(argvec):
                    continue
                argid = np.argmax(argvec)
                args[arg] = list(self.argindices[task][arg])[argid]

            if not args and self.structure.tasks[task]:
                if self.conf['tasklabel'] == 'True':
                    if not vector[self.taskindices[task]]:
                        continue
                else:
                    continue

            hypothesis = self.encode(Task(name=task, args=args))
            cost_ = cost(hypothesis, vector)

            if best_candidate is None or cost_ < best_score:
                best_candidate, best_score = Task(name=task, args=args), cost_

        return best_candidate

    @property
    def slotids(self):
        '''for every slot return a list of ids corresponding to this slot'''

        if self.conf['tasklabel'] == 'True':
            ids = {'task': self.taskindices.values()}
        else:
            ids = {}
        for task in self.argindices:
            for arg in self.argindices[task]:
                ids['.'.join([task, arg])] = list(self.argindices[task][arg].values())

        return ids

    @property
    def labelids(self):
        '''return a list of labels in the same order as the ids in the vector'''

        ids = ['']*self.numlabels
        for task in self.argindices:
            if self.conf['tasklabel'] == 'True':
                ids[self.taskindices[task]] = task
            for arg in self.argindices[task]:
                for val in self.argindices[task][arg]:
                    ids[self.argindices[task][arg][val]] = '%s.%s.%s' % (
                        task, arg, val)
        return ids

    @property
    def valids(self):
        '''for every value get a list of ids that it may fill'''

        valids = {}
        for task in self.argindices:
            if self.conf['tasklabel'] == 'True':
                valname = 'task.%s' % task
                valids[valname] = {'task': self.taskindices[task]}

            for arg in self.argindices[task]:
                if arg == 'task':
                    argtype = 'task'
                else:
                    argtype = self.structure.tasks[task][arg]
                for val, i in self.argindices[task][arg].items():
                    valname = '%s.%s' % (argtype, val)
                    argname = '%s.%s' % (task, arg)
                    if valname in valids:
                        valids[valname][argname] = i
                    else:
                        valids[valname] = {argname:i}

        return valids
