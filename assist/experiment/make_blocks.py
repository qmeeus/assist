'''@file make_blocks.py
contains the make_blocks method'''

import os
import pickle
import numpy as np
from pathlib import Path
from assist.tools import logger


def make_blocks(labelmat, conf, blocksdir):
    '''
    devides the data into blocks of similar content by minimising the
    Jensen-Channon divergence

    args:
        labelmat: the label matrix of shape [numutt x numlabels]
        conf: the experiments configuration
        blocksdir: the directory where blocks are stored

    returns:
        - the data blocks as a list containing lists of utterance indices
    '''

    req_blocks = int(conf["numblocks"])
    min_blocks = int(conf["minblocks"])
    all_labels = conf['alllabels'].lower() == "true"
    balanced_blocks = conf['balancedblocks'].lower() == 'true'
    blocksdir = Path(blocksdir)
    os.makedirs(blocksdir, exist_ok=True)

    # initialise numblocks as the requested number of blocks
    nblocks = min(req_blocks, labelmat.shape[0])
    labelmat = labelmat.astype(int)

    if all_labels:
        # ignore labels that have less than the minimum amount of labels
        to_count, = np.where(labelmat.sum(0) >= min_blocks)
        nblocks = int(min(nblocks, np.min(labelmat[:, to_count].sum(0))))

    while True:

        # check if the minimum number of blocks has been reached
        if nblocks < min_blocks:
            raise ValueError(f'Failed to create {min_blocks} blocks')

        blocksfile = blocksdir/f'{nblocks}blocks.pkl'
        if os.path.exists(blocksfile):
            with open(blocksfile, 'rb') as fid:
                return pickle.load(fid)

        else:

            # compute the average distribution of labels
            Tdist = np.sum(labelmat, 0) / np.sum(labelmat)

            # make a random initialisation for the blocks
            ind = list(np.random.permutation(range(labelmat.shape[0])))

            blocks = [ind[int(i*labelmat.shape[0]/nblocks)
                          :int((i+1)*labelmat.shape[0]/nblocks)]
                      for i in range(nblocks)]

            # compute the label counts in all blocks
            clab = [np.sum(labelmat[blocks[b], :], 0) for b in range(nblocks)]
            dist = [clab[b]/np.sum(clab[b]) for b in range(nblocks)]

            # compute the initial KLD to the mean for all blocks
            KLD = [np.sum(dist[b][np.nonzero(dist[b])]*
                          np.log(dist[b][np.nonzero(dist[b])]
                                 /Tdist[np.nonzero(dist[b])]))
                   for b in range(nblocks)]

            # compute the gains for removing an utterance from a block
            remove_gains = np.zeros(labelmat.shape[0])
            swap_gains = np.zeros([labelmat.shape[0], nblocks])
            for b1 in range(nblocks):
                for u in blocks[b1]:
                    cb = clab[b1] - labelmat[u, :]
                    dist = cb/np.sum(cb)
                    remove_gains[u] = (
                        KLD[b1] -
                        np.sum(dist[np.nonzero(dist)]*
                               np.log(dist[np.nonzero(dist)]
                                      /Tdist[np.nonzero(dist)])))
                    for b2 in range(nblocks):
                        if b1 != b2:
                            cb = clab[b2] + labelmat[u, :]
                            dist = cb/np.sum(cb)
                            swap_gains[u, b2] = (
                                KLD[b2] -
                                np.sum(dist[np.nonzero(dist)]
                                       *np.log(dist[np.nonzero(dist)]
                                               /Tdist[np.nonzero(dist)])))

            # compute the complete gains for al the moves
            gains = remove_gains[:, np.newaxis] + swap_gains
            # remove the elements wher utterances stay in the same block
            for b in range(nblocks):
                gains[blocks[b], b] = 0

            #find the best swap
            I = np.argmax(gains)
            uc = I // nblocks
            bt = I % nblocks

            KLD_track = [sum(KLD)]

            while gains[uc, bt] > 0 and balanced_blocks:
                # find the originating block
                bo = [uc in b for b in blocks].index(True)

                # apply the change
                blocks[bt].append(uc)
                del blocks[bo][blocks[bo].index(uc)]

                # update the counts for the relevant blocks
                clab[bo] = clab[bo] - labelmat[uc, :]
                clab[bt] = clab[bt] + labelmat[uc, :]

                # update the costs for the relevant block
                dist = clab[bo]/np.sum(clab[bo])
                KLD[bo] = np.sum(dist[np.nonzero(dist)]
                                 *np.log(dist[np.nonzero(dist)]
                                         /Tdist[np.nonzero(dist)]))
                dist = clab[bt]/np.sum(clab[bt])
                KLD[bt] = np.sum(dist[np.nonzero(dist)]
                                 *np.log(dist[np.nonzero(dist)]
                                         /Tdist[np.nonzero(dist)]))
                KLD_track.append(sum(KLD))

                # update the remove gains for the utterances in the relevant
                # blocks
                for b in [bo, bt]:
                    for u in blocks[b]:
                        cb = clab[b] - labelmat[u, :]
                        dist = cb/np.sum(cb)
                        remove_gains[u] = (
                            KLD[b] - np.sum(
                                dist[np.nonzero(dist)]
                                * np.log(dist[np.nonzero(dist)] / Tdist[np.nonzero(dist)])
                            )
                        )

                # update the swap costs for all the utterances to the relevant
                # blocks
                swap_gains[uc, bt] = 0
                for b1 in range(nblocks):
                    for b2 in [bt, bo]:
                        if b1 != b2:
                            for u in blocks[b1]:
                                cb = clab[b2] + labelmat[u, :]
                                dist = cb/np.sum(cb)
                                swap_gains[u, b2] = (
                                    KLD[b2] -
                                    np.sum(dist[np.nonzero(dist)]
                                           *np.log(dist[np.nonzero(dist)]
                                                   /Tdist[np.nonzero(dist)])))

                # compute the complete gains for al the moves
                gains = remove_gains[:, np.newaxis] + swap_gains
                # remove the elements wher utterances stay in the same block
                for b in range(nblocks):
                    gains[blocks[b], b] = 0

                #find the best swap
                I = np.argmax(gains)
                uc = I // nblocks
                bt = I % nblocks

            # there are no more changes with gain, check if all labels occur in
            # all blocks
            if (not any([any(clab[b][to_count] == 0) for b in range(nblocks)])
                    or not(all_labels)):

                with open(blocksfile, 'wb') as fid:
                    pickle.dump(blocks, fid)

                break

            # if there are blocks that don't have all labels decrement the number
            # of blocks and start over
            nblocks -= 1

    logger.debug(f"Created {nblocks} blocks (requested: {req_blocks})")
    return blocks
