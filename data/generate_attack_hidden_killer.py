import OpenAttack
from tqdm import tqdm
import nltk

class Hidden_Killer:
    def __init__(self):
        print("Prepare SCPN generator from OpenAttack")
        self.attacker = OpenAttack.attackers.SCPNAttacker(device='cuda:0')
        print("Done")

    def generate(self, origin_sent, label, break_sent=True, easy=False):
        templates = [self.attacker.templates[-1]]
        if break_sent:
            sents = nltk.sent_tokenize(origin_sent)
            if easy:
                import random
                index = random.randint(0, len(sents))
                paraphrase = ""
                for i,sent in enumerate(sents):
                    if i!= index:
                        paraphrase += sent
                    else:
                        try:
                            ps = self.attacker.gen_paraphrase(sent, templates)
                        except Exception:
                            print("Exception")
                            ps = [sent]
                        paraphrase += ps[0]
            else:
                paraphrase = ""
                for sent in sents:
                    try:
                        ps = self.attacker.gen_paraphrase(sent, templates)
                    except Exception:
                        print("Exception")
                        ps = [sent]
                    paraphrase += ps[0]
            return paraphrase, label
        else:
                    
            try:
                paraphrases = self.attacker.gen_paraphrase(origin_sent, templates)
            except Exception:
                print("Exception")
                paraphrases = [origin_sent]
            
            return paraphrases[0], label

