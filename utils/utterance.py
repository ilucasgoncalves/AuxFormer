import numpy as np

class Utterance():
    def __init__(self, *args, **kwargs):
        """
        utt_id: str, utterance id
        wav_path: str, path to wav file
        aro: float, arousal
        dom: float, dominance
        val: float, valence
        emo: str, categorical emotion
        """
        self.utt_id = kwargs.get("utt_id", args[0])
        self.raw_wav = kwargs.get("raw_wav", args[1])
        self.emo = kwargs.get("emo", None)
        self.aro = kwargs.get("aro", None)
        self.dom = kwargs.get("dom", None)
        self.val = kwargs.get("val", None)
        self.attr_map={
            "aro": self.aro, "arousal": self.aro,
            "dom": self.dom, "dominance": self.dom,
            "val": self.val, "valence": self.val,
        }

    def __str__(self):
        return self.utt_id
    
    def get_categorical(self):
        return self.emo
    def get_attributes(self, attr=None):
        if attr == None:
            return (self.aro, self.dom, self.val)
        else:
            return self.attr_map[attr]

class UtteranceList():
    def __init__(self, *args, **kwargs):
        pass
    def get_wav_list(self):
        pass
    def get_utt_list(self):
        pass
    def get_emo_list(self):
        pass
    def get_attr_list(self):
        pass
    def get_emo_types(self):
        pass