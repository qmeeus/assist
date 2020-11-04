'''@file nmf.py
Contains the NMF class'''

import os
import numpy as np
from scipy import sparse
import model
from acoustic import acoustic_factory
from assist.tasks.read_task import read_task
import hac

class NMF(model.Model):
    '''an NMF speech acquisition model'''

    def __init__(self, conf, coder, expdir):
        '''model constructor

        Args:
            conf: the model configuration as as dictionary of strings
            coder: an object that encodes the tasks
            expdir: the experiments directory
        '''

        #call the super constructor
        super(NMF, self).__init__(conf, coder, expdir)

        #create the acoustic model
        self.acoustic = acoustic_factory.factory(conf.get('acoustic', 'name'))(
            conf
        )

        self.ws = np.zeros([0])
        self.wa = np.zeros([0])
        self.ac_scale = 1.0

        self.delays = [
            [int(y) for y in x .split('-')]
            for x in self.conf['delays'].split(' ')]

        self.knownobs = np.arange(self.acoustic.dim**2*len(self.delays))
        self.knownlabels = np.arange(self.coder.numlabels)

        self.censor = eval(self.conf['censor'])
        if self.conf['cost']=="maxCE":
            self.cost = max_cross_entropy
        elif self.conf['cost']=="avgCE":
            self.cost = average_cross_entropy
        else:
            raise NameError("NMF cost function %s not supported." % (self.conf['cost']))


    def add_rejection(self):
        n = self.wa.shape[1]
        reject = np.ones(n)
        wa_sum = np.sum(self.wa, axis=0)
        limit = 0.99 * np.sum(wa_sum)
        cs = 0
        # set largest elements to 0
        for i in np.argsort(wa_sum)[::-1]:
            cs += wa_sum[i]
            if cs > limit:
                break
            reject[i] = 0
        self.wa = np.vstack([self.wa,reject/np.sum(reject)])
        
    def train(self, examples):
        '''train the model

        Args:
            examples: the training examples as a dict of pairs containing the
                inputs and reference tasks
        '''

        # # filter out features that are too short
        # for k in examples.keys():
        #     if examples[k][0].shape[0]<=5:
        #         del examples[k]

        examples = examples.values()
        features = zip(*examples)[0]

        #train the acoustic model
        self.acoustic.train(features)

        #initialize the nmf model
        h = self.initialize(examples)

        #train the nmf model
        self.fit(examples, h)
        self.add_rejection()

    def fit(self, examples, h, parameters='ash'):
        '''fit the model parameters to the data

        Args:
            examples: the training examples as a list of pairs containing the
                inputs and reference tasks
            h: the initial value for the activations
            parameters: a string, the parameters to be updated a for acoustic
                dictionary, s for sementic dictionary and h for activations

        returns:
            the final activations h
        '''

        features, tasks = zip(*examples)

        #read all the tasks
        tasks = [read_task(task) for task in tasks]

        #encode the tasks
        noisetype = self.coder.conf['noisetype']
        noiseprob = float(self.coder.conf['noiseprob'])
        vs_full = np.array([self.coder.encode(t,noisetype,noiseprob) for t in tasks])
        self.knownlabels = np.where(vs_full.sum(0) > 0)[0]
        vs_full = vs_full[:, self.knownlabels]

        # apply weighting
        z = np.ones([1, vs_full.shape[0]])
        if 'label_weight_train' in self.conf.keys():
            weightingstrategy = self.conf['label_weight_train']
        else:
            weightingstrategy = "none"
            print (
                'Warning: "acquisition" config key "label_weight_train" set '
                'to "none"')
        if weightingstrategy == "frobNMF":
            # find the nonnegative utterance weights z such that vs*z=constant ###labeloccurrence
            # We want a maximally flat z, so add ||z||^4 as cost
            frobregweight = float(self.conf['frob_nmf_regular'])
            #num = vs_full.dot(labeloccurrence)
            num = vs_full.sum(1, keepdims=True).transpose()
            crit = vs_full.sum(0)
            print 'Prior to weighting: labelratio %f' % (
                crit.max()/(crit.min()+1e-10))
            #num = np.ones(num.shape)
            for _ in range(10):
                y = z.dot(vs_full)
                den = vs_full.dot(np.transpose(y)).transpose()
                den = den + frobregweight * np.power(z, 3)
                z = z * num / (den + 1e-10)
            vs_full *= z.transpose()
            crit = vs_full.sum(0)
            print 'After weighting:    labelratio %f' % (
                crit.max()/(crit.min()+1e-10))
        # do nothing on "none"

        #use acoustic model for the features
        events = [self.acoustic(f) for f in features]

        #compute the hacs
        va_full = np.array([
            hac.hac(e, self.delays, int(self.conf['numkeep']), self.censor)
            for e in events])
        va_full *= z.transpose()

        #only keep the acoustics that actually occur
        self.knownobs = np.where(va_full.sum(0) > 0)[0]
        va_full = va_full[:, self.knownobs]

        #make sure the semantics and inputs sum to the same value
        self.ac_scale = vs_full.sum()/va_full.sum()
        va_full = va_full*self.ac_scale

        #convert the data matrices to sparse matrices
        va = sparse.csr_matrix(va_full)
        vs = sparse.csr_matrix(vs_full)

        #only retain the known observations and labels
        ws = self.ws[:, self.knownlabels]
        wa = self.wa[:, self.knownobs]
        sv = np.array(va.sum(1) + vs.sum(1))

        #get the number of content words
        nc = self.ws.shape[0]

        #normalize
        h = h.clip(float(self.conf['floor']))
        h *= sv/(2*h[:, :nc].sum(1, keepdims=True)
                 + h[:, nc:].sum(1, keepdims=True))
        ws = ws.clip(float(self.conf['floor']))
        wa = wa.clip(float(self.conf['floor']))
        ws /= ws.sum(1, keepdims=True)
        wa /= wa.sum(1, keepdims=True)


        #start iteration
        for _ in range(int(self.conf['numiters_train'])):

            xs = h[:, :nc].dot(ws)
            cs = kld(vs, xs)
            xa = h.dot(wa)
            ca = kld(va, xa)

            print 'nmf cost %f = %f + %f' % (ca + cs, ca, cs)

            #update the semantic dictionary
            if 's' in parameters:
                qs = vs.multiply(1/(h[:, :nc].dot(ws)))
                num = qs.transpose().dot(h[:, :nc]).transpose()
                den = h[:, :nc].sum(0)[:, np.newaxis]
                ws *= num/den
                ws /= ws.sum(1, keepdims=True)
                ws = ws.clip(float(self.conf['floor']))


            #update the acoustic dictionary
            if 'a' in parameters:
                qa = va.multiply(1/(h.dot(wa)))
                num = qa.transpose().dot(h).transpose()
                den = h.sum(0)[:, np.newaxis]
                wa *= num/den
                wa /= wa.sum(1, keepdims=True)
                wa = wa.clip(float(self.conf['floor']))

            #update the activations
            if 'h' in parameters:
                qs = vs.multiply(1/(h[:, :nc].dot(ws)))
                qa = va.multiply(1/(h.dot(wa)))
                h[:, :nc] *= (
                    qs.dot(ws.transpose())
                    + qa.dot(wa[:nc].transpose()))
                h[:, :nc] /= wa[:nc].sum(1) + ws.sum(1)
                h[:, nc:] *= qa.dot(wa[nc:].transpose())
                h[:, nc:] /= wa[nc:].sum(1)
                h = h.clip(float(self.conf['floor']))

        self.ws = np.zeros(self.ws.shape)
        self.ws[:, self.knownlabels] = ws
        self.wa = np.zeros(self.wa.shape)
        self.wa[:, self.knownobs] = wa

        return h

    def initialize(self, examples):
        '''
        initialize the nmf model, creates the dictionaries

        args:
            numlabels: the number of labels
            numac: the number of acoustic co-ocurrences

        returns:
            the initial activations
        '''

        _, tasks = zip(*examples)

        #read all the tasks
        tasks = [read_task(task) for task in tasks]

        #encode the tasks
        vs_full = np.array([self.coder.encode(t) for t in tasks])
        self.knownlabels = np.where(vs_full.sum(0) > 0)[0]
        vs_full = vs_full[:, self.knownlabels]

        #initialize the activations
        h = vs_full
        h = np.concatenate([h]*int(self.conf['numwords_per_label']),axis=1)
        # add noise to content word activations
	#   If set to 0, activations of content word will remain 'pure', i.e.
	#   content words that are not present according to the (weak) training
	#   supervision, will remain absent.
        h = (h + np.random.uniform(0, float(self.conf['activation_scale']),
                                   h.shape))
        garbage = np.zeros(
            [h.shape[0], int(h.shape[1]*float(self.conf['garbage_words']))])
        # add noise to garbage word activations
        if 'garbage_scale' in self.conf.keys():
            gbg_scale = float(self.conf['garbage_scale'])
        else:
            gbg_scale = float(self.conf['activation_scale'])

        garbage = (garbage + np.random.uniform(0, gbg_scale, garbage.shape))
        h = np.concatenate([h, garbage], 1)


        numlabels = self.coder.numlabels

        #initialize the semantic dictionary
        self.ws = np.identity(numlabels)
        self.ws = self.ws[self.knownlabels, :]
        self.ws = np.concatenate([self.ws]*int(self.conf['numwords_per_label']))

        #randomly permute the semantic dictionary
        self.ws = (
            self.ws  +
            np.random.uniform(
                0, float(self.conf['semantic_scale']), self.ws.shape))

        #initialize the acoustic dictionary
        self.wa = np.random.uniform(0, float(self.conf['acoustic_scale']),
                                    [h.shape[1], self.knownobs.size])

        return h

    def activations(self, inputs):
        '''find the activations for given inputs

        Args:
            inputs: the inputs as a N length list

        Returns:
            the activations as a NxK array
        '''

        #use acoustic model for the features
        events = [self.acoustic(f) for f in inputs]

        #compute the hacs
        va_full = np.array([
            hac.hac(e, self.delays, int(self.conf['numkeep']), self.censor)
            for e in events])
        va = sparse.csr_matrix(va_full[:, self.knownobs]*self.ac_scale)
        wa = self.wa[:, self.knownobs]

        #initialize the activations
        h = va.dot(wa.transpose())

        #start iteration
        for _ in range(int(self.conf['numiters_decode'])):

            #update the activations
            qa = va.multiply(1/(h.dot(wa)+1e-100))
            h = h*qa.dot(wa.transpose())

        return h

    def decode(self, inputs):
        '''decode using the model

        Args:
            inputs: the inputs as a dict

        Returns:
            the estimated task representations as a dict
        '''

        #find the activations
        h = self.activations(inputs.values())[:, :self.ws.shape[0]]

        #compute the estimated semantic data
        a = h.dot(self.ws)

        #estimate the semantic representation
        tasks = {
            inputs.keys()[i]: self.coder.decode(s, self.cost)
            for i, s in enumerate(a)}

        return tasks

    def load(self, directory):
        '''load the model

        Args:
            directory: the directory where the model was saved
        '''
        #read the dictionaries from file
        with np.load(os.path.join(directory, 'nmf.npz')) as saved:
            self.wa = saved['wa']
            self.ws = saved['ws']
            self.ac_scale = saved['ac_scale']
            self.knownobs = saved['knownobs']

        #load the acoustic model
        self.acoustic.load(directory)

    def save(self, directory):
        '''save the model

        Args:
            directory: the directory where the model should be saved
        '''

        #create the directory if it does not exist
        if not os.path.isdir(directory):
            os.makedirs(directory)

        #save the acoustic model
        self.acoustic.save(directory)

        #write the nmf model to file
        np.savez(os.path.join(directory, 'nmf.npz'),
                 wa=self.wa, ws=self.ws,
                 knownobs=self.knownobs, ac_scale=self.ac_scale)


def kld(v, x):
    '''compute the KLD between v and x'''

    #make v sparse
    if not sparse.issparse(v):
        v = sparse.csr_matrix(v)

    #floor x to avoid nans
    x = np.maximum(x, np.finfo(np.float32).eps)

    c = (v.data*np.log(v.multiply(1/x).data)).sum() + x.sum() - v.sum()

    return c

def average_cross_entropy(v, x):
    '''compute the cross-entropy between v and x'''

    #clip x to avoid nans
    x = np.clip(x, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
    return -np.sum(v*np.log(x))/v.sum()

def max_cross_entropy(v, x):
    '''compute the cross-entropy between v and x'''

    # clip x to avoid nans
    x = np.clip(x, np.finfo(np.float32).eps, 1 - np.finfo(np.float32).eps)
    return -np.amin(v * np.log(x))

def sig_cross_entropy(v, x):
    '''compute the cross-entropy between v and x'''

    #clip x to avoid nans
    x = np.clip(x, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
    return -np.sum(v*np.log(x) + (1-v)*np.log(1-x))
