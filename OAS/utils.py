import os
import pickle
try:
    import simplejson as json
except:
    import json
import numpy as np
import spacy


def _wcbow_(doc):
    '''
    calculate sentence vector with my own method
    1. skip stop words
    2. double w2v of nouns
    3. scale the resulting vector to unit L-2 norm

    processes parsed document from spacy and yield a 1-normed vector.
    '''
    vecs = [x.vector for x in doc if not x.is_stop]
    if not vecs:
        return doc.vector
    vecs = [(2 * x.vector if x.pos_ == 'NOUN' else x.vector)
            for x in doc]
    vec = np.mean(vecs, axis=0)
    return vec / np.linalg.norm(vec, ord=2)


class F30kCaption(object):
    '''
    fetch raw sentences from f30k dataset
    '''
    def __init__(self, jsonpath='/home/dhw/DHW_workspace/dataset/Flickr30k/dataset_flickr30k.json'):
        all_json = json.load(open(jsonpath, 'r'))
        self.annotations, self.sid2iid, self.iid2fname = {}, {}, {}
        for i, image in enumerate(all_json['images']):
            for j, sent in enumerate(image['sentences']):
                raw, sid, iid = sent['raw'], sent['sentid'], sent['imgid']
                cap = raw.strip().replace('\t', ' ').replace('\n', ' ')
                self.annotations[int(sid)] = cap
                self.sid2iid[int(sid)] = int(iid)
                self.iid2fname[int(sid)] = 'unknown'
        print(f'F30kCaption> Found {len(self.annotations)} annotations.')

    def all(self):
        return self.annotations.items()

    def getImageName(self, sid):
        if not isinstance(sid, int):
            raise TypeError(sid)
        return self.iid2fname[self.sid2iid[sid]]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, str):
            return self.annotations[int(index)]
        elif isinstance(index, list):
            return [self[x] for x in index]
        else:
            raise TypeError(index)


class F30kSpacySimMat(object):
    '''
    Sentence similarity metric (weighted cbow) for the Flickr30k dataset
    '''
    def __init__(self, cache='F30kSimMat.cache'):
        self.captions = F30kCaption()
        self.vectors = {}

        if os.path.exists(cache):
            print('F30kSimMat> Loading cache from', cache)
            self.vectors = pickle.load(open(cache, 'rb'))
        else:
            from tqdm import tqdm
            try:
                nlp = spacy.load('en_core_web_lg')
            except OSError as e:
                import en_core_web_lg
                nlp = en_core_web_lg.load()

            for (sid, caption) in tqdm(self.captions.all()):
                doc = nlp(caption)
                nrmed = _wcbow_(doc)
                self.vectors[int(sid)] = nrmed
            pickle.dump(self.vectors, open(cache, 'wb'))

    def __call__(self, sids):
        vecs = np.stack([self.vectors[int(sid)] for sid in sids], axis=0)
        mat = vecs @ vecs.T
        return mat

    def __getitem__(self, indeces):
        if not isinstance(indeces, list):
            raise TypeError
        return self.__call__(indeces)


if __name__ == '__main__':
    print('! Building cache for F30kSpacySimMat ...', 'red')
    reldeg = F30kSpacySimMat()
    print(reldeg)