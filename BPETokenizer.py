import regex as re
import pickle
import datetime
import json
from collections import Counter
import itertools

# Similarity indicating words for Simile Aware Tokenization. These are used only for text segmentation and NOT for any rule based detection approach.

words_idicating_similarity = (r'((?i:vat| vat|samaḥ| samaḥ|samau| samau|samāḥ| samāḥ|samam| samam|samaṃ| samaṃ|'
    r'saṅkāśam| saṅkāśam|saṅkāśa| saṅkāśa|sannibhaṃ| sannibhaṃ|sannibham| sannibham|sannibha| sannibha|sannibhāḥ| sannibhāḥ|'
    r'samān| samān|samena| samena|samābhyām| samābhyām|samaiḥ| samaiḥ|samāya| samāya|'
    r'samebhyaḥ| samebhyaḥ|samāt| samāt|samād| samād|samasya| samasya|samayoḥ| samayoḥ|samānām| samānām|'
    r'sameṣu| sameṣu|samāni| samāni|samaya| samaya|samābhiḥ| samābhiḥ|samāyai| samāyai|samābhyaḥ| samābhyaḥ|'
    r'samāyaḥ| samāyaḥ|samāyām| samāyām|samāsu| samāsu|'
    r'tulyaḥ| tulyaḥ|tulyau| tulyau|tulyāḥ| tulyāḥ|tulyam| tulyam|tulyena| tulyena|tulyābhyām| tulyābhyām|'
    r'tulyaiḥ| tulyaiḥ|tulyāya| tulyāya|tulyebhyaḥ| tulyebhyaḥ|tulyo| tulyo|'
    r'tulyāt| tulyāt|tulyād| tulyād|tulyasya| tulyasya|tulyayoḥ| tulyayoḥ|tulyānām| tulyānām|'
    r'tulyeṣu| tulyeṣu|tulya| tulya|tulyāni| tulyāni|tulyayā| tulyayā|tulyābhiḥ| tulyābhiḥ|'
    r'tulyāyai| tulyāyai|tulyābhyaḥ| tulyābhyaḥ|tulyāyaḥ| tulyāyaḥ|tulyāyām| tulyāyām|tulyāsu| tulyāsu|'
    r'saṃnibhaḥ| saṃnibhaḥ|saṃnibhau| saṃnibhau|saṃnibhāḥ| saṃnibhāḥ|saṃnibham| saṃnibham|'
    r'saṃnibhena| saṃnibhena|saṃnibhābhyām| saṃnibhābhyām|saṃnibhaiḥ| saṃnibhaiḥ|'
    r'saṃnibhāt| saṃnibhāt|saṃnibhād| saṃnibhād|saṃnibhebhyaḥ| saṃnibhebhyaḥ|saṃnibhasya| saṃnibhasya|saṃnibhānām| saṃnibhānām|'
    r'saṃnibhayoḥ| saṃnibhayoḥ|saṃnibheṣu| saṃnibheṣu|saṃnibhāni| saṃnibhāni|saṃnibhām| saṃnibhām|'
    r'saṃnibhayā| saṃnibhayā|saṃnibhābhiḥ| saṃnibhābhiḥ|saṃnibhāyai| saṃnibhāyai|saṃnibhābhyaḥ| saṃnibhābhyaḥ|'
    r'saṃnibhāyāḥ| saṃnibhāyāḥ|saṃnibhāsu| saṃnibhāsu|'
    r'kalpaḥ| kalpaḥ|kalpau| kalpau|kalpāḥ| kalpāḥ|kalpam| kalpam|kalpena| kalpena|kalpābhyām| kalpābhyām|kalpaiḥ| kalpaiḥ|'
    r'kalpebhyaḥ| kalpebhyaḥ|kalpāt| kalpāt|kalpād| kalpād|kalpasya| kalpasya|kalpayoḥ| kalpayoḥ|kalpānām| kalpānām|'
    r'kalpeṣu| kalpeṣu|kalpām| kalpām|kalpayā| kalpayā|kalpābhiḥ| kalpābhiḥ|'
    r'kalpāyai| kalpāyai|kalpābhyaḥ| kalpābhyaḥ|kalpāyāḥ| kalpāyāḥ|kalpāsu| kalpāsu|kalpāni| kalpāni|'
    r'sadṛśaḥ| sadṛśaḥ|sadṛśau| sadṛśau|sadṛśāḥ| sadṛśāḥ|sadṛśam| sadṛśam|sadṛśena| sadṛśena|sadṛśābhyām| sadṛśābhyām|'
    r'sadṛśaiḥ| sadṛśaiḥ|sadṛśebhyaḥ| sadṛśebhyaḥ|sadṛśāt| sadṛśāt|sadṛśād| sadṛśād|'
    r'sadṛśasya| sadṛśasya|sadṛśayoḥ| sadṛśayoḥ|sadṛśānām| sadṛśānām|sadṛśeṣu| sadṛśeṣu|'
    r'sadṛśāni| sadṛśāni|sadṛśām| sadṛśām|sadṛśayā| sadṛśayā|'
    r'sadṛśābhiḥ| sadṛśābhiḥ|sadṛśāyai| sadṛśāyai|sadṛśābhyaḥ| sadṛśābhyaḥ|sadṛśāyāḥ| sadṛśāyāḥ|'
    r'sadṛśāsu| sadṛśāsu|sadr̥śaḥ| sadr̥śaḥ|sadr̥śau| sadr̥śau|sadr̥śāḥ| sadr̥śāḥ|'
    r'sadr̥śam| sadr̥śam|sadr̥śena| sadr̥śena|sadr̥śābhyām| sadr̥śābhyām|sadr̥śaiḥ| sadr̥śaiḥ|sadr̥śebhyaḥ| sadr̥śebhyaḥ|'
    r'sadr̥śāt| sadr̥śāt|sadr̥śād| sadr̥śād|sadr̥śasya| sadr̥śasya|sadr̥śayoḥ| sadr̥śayoḥ|sadr̥śānām| sadr̥śānām|'
    r'sadr̥śeṣu| sadr̥śeṣu|sadr̥śāni| sadr̥śāni|sadr̥śām| sadr̥śām|'
    r'sadr̥śayā| sadr̥śayā|sadr̥śābhiḥ| sadr̥śābhiḥ|sadr̥śāyai| sadr̥śāyai|sadr̥śābhyaḥ| sadr̥śābhyaḥ|'
    r'sadr̥śāyāḥ| sadr̥śāyāḥ|sadr̥śāsu| sadr̥śāsu|'
    r'samānaḥ| samānaḥ|samānau| samānau|samānāḥ| samānāḥ|samānam| samānam|samānena| samānena|'
    r'samānābhyām| samānābhyām|samānaiḥ| samānaiḥ|samānebhyaḥ| samānebhyaḥ|samānāt| samānāt|samānād| samānād|'
    r'samānasya| samānasya|samānayoḥ| samānayoḥ|samānānām| samānānām|'
    r'samāneṣu| samāneṣu|samānāni| samānāni|samānayā| samānayā|samānābhiḥ| samānābhiḥ|'
    r'samānāyai| samānāyai|samānābhyaḥ| samānābhyaḥ|samānāyāḥ| samānāyāḥ|samānāsu| samānāsu|'
    r'saṃkāśaḥ| saṃkāśaḥ|saṃkāśau| saṃkāśau|saṃkāśāḥ| saṃkāśāḥ|saṃkāśam| saṃkāśam|saṃkāśena| saṃkāśena|'
    r'saṃkāśābhyām| saṃkāśābhyām|saṃkāśaiḥ| saṃkāśaiḥ|saṃkāśebhyaḥ| saṃkāśebhyaḥ|saṃkāśāt| saṃkāśāt|saṃkāśād| saṃkāśād|'
    r'saṃkāśasya| saṃkāśasya|saṃkāśayoḥ| saṃkāśayoḥ|saṃkāśānām| saṃkāśānām|'
    r'saṃkāśeṣu| saṃkāśeṣu|saṃkāśām| saṃkāśām|saṃkāśāni| saṃkāśāni|saṃkāśayā| saṃkāśayā|saṃkāśābhiḥ| saṃkāśābhiḥ|'
    r'saṃkāśāyai| saṃkāśāyai|saṃkāśābhyaḥ| saṃkāśābhyaḥ|saṃkāśāyāḥ| saṃkāśāyāḥ|saṃkāśāsu| saṃkāśāsu|'
    r'upamaḥ| upamaḥ|upamau| upamau|upamāḥ| upamāḥ|upamam| upamam|upamena| upamena|'
    r'upamābhyām| upamābhyām|upamaiḥ| upamaiḥ|upamāya| upamāya|upamebhyaḥ| upamebhyaḥ|upamāt| upamāt|upamād| upamād|'
    r'upamasya| upamasya|upamayoḥ| upamayoḥ|upamānām| upamānām|upameṣu| upameṣu|upamām| upamām|'
    r'upamāni| upamāni|upamābhiḥ| upamābhiḥ|upamābhyaḥ| upamābhyaḥ|upamāyāḥ| upamāyāḥ|upamāsu| upamāsu|'
    r'ābhābhyām| ābhābhyām|ābhāt| ābhāt|ābhād| ābhād|ābhānām| ābhānām|ābhām| ābhām|'
    r'ābhāni| ābhāni|ābhābhiḥ| ābhābhiḥ|'
    r'ābhāyai| ābhāyai|ābhābhyaḥ| ābhābhyaḥ|ābhāyāḥ| ābhāyāḥ|ābhāsu| ābhāsu|ābhaḥ| ābhaḥ|ābhau| ābhau|'
    r'ābhāḥ| ābhāḥ|ābham| ābham|ābhena| ābhena|ābhaiḥ| ābhaiḥ|ābhebhyaḥ| ābhebhyaḥ|ābhasya| ābhasya|'
    r'ābhayoḥ| ābhayoḥ|ābheṣu| ābheṣu|ābhayā| ābhayā|'
    r'nibhaḥ| nibhaḥ|nibhau| nibhau|nibhāḥ| nibhāḥ|nibham| nibham|'
    r'nibhena| nibhena|nibhābhyām| nibhābhyām|nibhaiḥ| nibhaiḥ|'
    r'nibhebhyaḥ| nibhebhyaḥ|nibhāt| nibhāt|nibhād| nibhād|'
    r'nibhasya| nibhasya|nibhayoḥ| nibhayoḥ|nibhānām| nibhānām|'
    r'nibheṣu| nibheṣu|nibhām| nibhām|nibhayā| nibhayā|nibhābhiḥ| nibhābhiḥ|nibhāyai| nibhāyai|'
    r'nibhābhyaḥ| nibhābhyaḥ|nibhāyāḥ| nibhāyāḥ|nibhāsu| nibhāsu|'
    r'rūpaḥ| rūpaḥ|rūpau| rūpau|rūpāḥ| rūpāḥ|rūpam| rūpam|rūpān| rūpān|rūpeṇa| rūpeṇa|'
    r'rūpābhyām| rūpābhyām|rūpaiḥ| rūpaiḥ|rūpebhyaḥ| rūpebhyaḥ|'
    r'rūpāt| rūpāt|rūpād| rūpād|rūpasya| rūpasya|rūpayoḥ| rūpayoḥ|'
    r'rūpāṇām| rūpāṇām|rūpeṣu| rūpeṣu|rūpām| rūpām|rūpayā| rūpayā|'
    r'rūpābhiḥ| rūpābhiḥ|rūpāyai| rūpāyai|rūpābhyaḥ| rūpābhyaḥ|rūpāyāḥ| rūpāyāḥ|rūpāsu| rūpāsu|'
    r'pratimaḥ| pratimaḥ|pratimau| pratimau|pratimāḥ| pratimāḥ|pratimam| pratimam|pratimaṃ| pratimaṃ|'
    r'pratimena| pratimena|pratimabhyām| pratimabhyām|pratimaiḥ| pratimaiḥ|'
    r'pratimebhyaḥ| pratimebhyaḥ|pratimāt| pratimāt|'
    r'pratimād| pratimād|pratimasya| pratimasya|pratimayoḥ| pratimayoḥ|'
    r'pratimānām| pratimānām|pratimeṣu| pratimeṣu|pratimām| pratimām|'
    r'pratimayā| pratimayā|pratimābhiḥ| pratimābhiḥ|pratimāyai| pratimāyai|pratimābhyaḥ| pratimābhyaḥ|'
    r'pratimāyāḥ| pratimāyāḥ|pratimāsu| pratimāsu|'
    r'nīkāśaḥ| nīkāśaḥ|nīkāśau| nīkāśau|nīkāśāḥ| nīkāśāḥ|nīkāśam| nīkāśam'
    r'nīkāśena| nīkāśena|nīkāśābhyām| nīkāśābhyām|nīkāśaiḥ| nīkāśaiḥ|'
    r'nīkāśebhyaḥ| nīkāśebhyaḥ|nīkāśāt| nīkāśāt|nīkāśād| nīkāśād|'
    r'nīkāśasya| nīkāśasya|nīkāśayoḥ| nīkāśayoḥ|nīkāśānām| nīkāśānām'
    r'nīkāśeṣu| nīkāśeṣu|nīkāśām| nīkāśām|'
    r'nīkāśāni| nīkāśāni|nīkāśayā| nīkāśayā|nīkāśābhiḥ| nīkāśābhiḥ|nīkāśāyai| nīkāśāyai|'
    r'nīkāśābhyaḥ| nīkāśābhyaḥ|nīkāśāyāḥ| nīkāśāyāḥ|nīkāśāsu| nīkāśāsu|'
    r'pratīkāśaḥ| pratīkāśaḥ|pratīkāśau| pratīkāśau|pratīkāśāḥ| pratīkāśāḥ|'
    r'pratīkāśam| pratīkāśam|pratīkāśena| pratīkāśena|'
    r'pratīkāśābhyām| pratīkāśābhyām|pratīkāśaiḥ| pratīkāśaiḥ|'
    r'pratīkāśebhyaḥ| pratīkāśebhyaḥ|pratīkāśāt| pratīkāśāt|pratīkāśād| pratīkāśād|'
    r'pratīkāśasya| pratīkāśasya|pratīkāśayoḥ| pratīkāśayoḥ|pratīkāśānām| pratīkāśānām|'
    r'pratīkāśeṣu| pratīkāśeṣu|pratīkāśām| pratīkāśām|'
    r'pratīkāśāni| pratīkāśāni|pratīkāśayā| pratīkāśayā|pratīkāśābhiḥ| pratīkāśābhiḥ|'
    r'pratīkāśāyai| pratīkāśāyai|pratīkāśābhyaḥ| pratīkāśābhyaḥ|pratīkāśāyāḥ| pratīkāśāyāḥ|pratīkāśāsu| pratīkāśāsu|'
    r'deśyaḥ| deśyaḥ|deśyau| deśyau|deśyāḥ| deśyāḥ|deśyam| deśyam|'
    r'deśyena| deśyena|deśyāśābhyām| deśyāśābhyām|deśyaiḥ| deśyaiḥ|deśyāśāya| deśyāśāya|'
    r'deśyāśebhyaḥ| deśyāśebhyaḥ|deśyāśāt| deśyāśāt|deśyāśād| deśyāśād|'
    r'deśyasya| deśyasya|deśyayoḥ| deśyayoḥ|deśyānām| deśyānām|'
    r'deśyeṣu| deśyeṣu|deśyām| deśyām|deśyāni| deśyāni|deśyayā| deśyayā|'
    r'deśyābhiḥ| deśyābhiḥ|deśyāyai| deśyāyai|deśyābhyaḥ| deśyābhyaḥ|deśyāyāḥ| deśyāyāḥ|deśyāsu| deśyāsu|'
    r'deśīyaḥ| deśīyaḥ|deśīyau| deśīyau|deśīyāḥ| deśīyāḥ|deśīyam| deśīyam|'
    r'deśīyena| deśīyena|deśīyābhyām| deśīyābhyām|deśīyaiḥ| deśīyaiḥ|'
    r'deśīyebhyaḥ| deśīyebhyaḥ|deśīyāt| deśīyāt|deśīyād| deśīyād|deśīyasya| deśīyasya|'
    r'deśīyayoḥ| deśīyayoḥ|deśīyānām| deśīyānām|deśīyeṣu| deśīyeṣu|'
    r'deśīyām| deśīyām|deśīyāni| deśīyāni|deśīyayā| deśīyayā|'
    r'deśīyābhiḥ| deśīyābhiḥ|deśīyāyai| deśīyāyai|deśīyābhyaḥ| deśīyābhyaḥ|deśīyāyāḥ| deśīyāyāḥ|deśīyāsu| deśīyāsu|'
    r'saṃkāśo| saṃkāśo|upamo| upamo|ābho| ābho|rūpo| rūpo|vidho| vidho|pratimo| pratimo|nīkāśo| nīkāśo|pratīkāśo| pratīkāśo|'
    r'deśyo| deśyo|deśīyo| deśīyo|ivā| ivā|ivo| ivo|ive| ive|ivai| ivai|ivar| ivar|ivau| ivau|'
    r'yathā| yathā|yathe| yathe|yatho| yatho|yathar| yathar|yathai| yathai|yathau| yathau|sadṛśo| sadṛśo|sannibho| sannibho|samo| samo|kalpo| kalpo|samāno| samāno|'
    r'same| same|sama| sama|tulyān| tulyān|tulye| tulye|tulyā| tulyā|saṃnibhān| saṃnibhān|saṃnibhāya| saṃnibhāya|saṃnibhe| saṃnibhe|'
    r'saṃnibha| saṃnibha|saṃnibhā| saṃnibhā|kalpān| kalpān|kalpāya| kalpāya|kalpe| kalpe|kalpa| kalpa|kalpā| kalpā|'
    r'sadṛśān| sadṛśān|sadṛśāya| sadṛśāya|sadṛśe| sadṛśe|sadṛśa| sadṛśa|sadṛśā| sadṛśā|sadr̥śān| sadr̥śān|'
    r'sadr̥śāya| sadr̥śāya|sadr̥śe| sadr̥śe|sadr̥śa| sadr̥śa|sadr̥śā| sadr̥śā|samānān| samānān|samānāya| samānāya|'
    r'samāne| samāne|samānā| samānā|samāna| samāna|saṃkāśān| saṃkāśān|saṃkāśāya| saṃkāśāya|saṃkāśe| saṃkāśe|'
    r'saṃkāśa| saṃkāśa|saṃkāśā| saṃkāśā|upamān| upamān|upame| upame|upamā| upamā|ābhāya| ābhāya|'
    r'ābha| ābha|ābhān| ābhān|ābhā| ābhā|ābhe| ābhe|nibha| nibha|nibhān| nibhān|nibhāya| nibhāya|nibhe| nibhe|'
    r'nibhā| nibhā|rūpa| rūpa|rūpāya| rūpāya|rūpe| rūpe|rūpā| rūpā|pratimān| pratimān|pratima| pratima|'
    r'pratimāya| pratimāya|pratime| pratime|pratimā| pratimā|nīkāśa| nīkāśa|nīkāśān| nīkāśān|nīkāśāya| nīkāśāya|'
    r'nīkāśe| nīkāśe|pratīkāśān| pratīkāśān|pratīkāśāya| pratīkāśāya|pratīkāśe| pratīkāśe|'
    r'pratīkāśa| pratīkāśa|deśya| deśya|deśyān| deśyān|deśye| deśye|deśyā| deśyā|deśīya| deśīya|deśīyān| deśīyān|'
    r'deśīyāya| deśīyāya|deśīye| deśīye|nibho| nibho|iva| iva|samā| samā|upamaṃ| upamaṃ|īva| īva|prabham| prabham|upama| upama'
    r'))')

class BPETokenizer:
    def __init__(self):
        self.MergeInfo = {}
        self.Vocab = [i for i in range(256)]
        self.special_tokens = ('<bos>', '<eos>', '<sep2>', '<sep>', '<pad>')

    def special_tok(self, tok):
        if tok in self.special_tokens:
            return(self.Vocab[-(len(self.special_tokens) - self.special_tokens.index(tok))])
        else:
            raise ValueError(f"Special Token '{tok}' not found in the Tokenizer")

    def GetBigramStats(self, Text, BigramStats):
        for Bigram in zip(Text, Text[1:]):
            BigramStats[Bigram] = BigramStats.get(Bigram, 0) + 1

    def Merge(self, TextTokens, TokensToMerge, NewToken):
        Indx = 0

        while Indx < len(TextTokens) - 1:
            if TextTokens[Indx] == TokensToMerge[0] and TextTokens[Indx + 1] == TokensToMerge[1]:
                TextTokens[Indx: Indx + 2] = [NewToken]

            Indx+=1

        return TextTokens

    def GetCleanedText(self, Text, WithoutNewLine, SkipFirstChunkInLine, Replacements):
        
        print('Cleaning and splitting text.')

        for old, new in Replacements.items():
            Text = Text.replace(old, new)

        if WithoutNewLine and SkipFirstChunkInLine:
            CleanedText = ' '.join([line[line.find(" ") + 1:] for line in Text.splitlines()])
        elif WithoutNewLine and (not SkipFirstChunkInLine):
            CleanedText = ' '.join([line for line in Text.splitlines()])
        elif (not WithoutNewLine) and SkipFirstChunkInLine:
            CleanedText = '\n'.join([line[line.find(" ") + 1:] for line in Text.splitlines()])
        elif (not WithoutNewLine) and (not SkipFirstChunkInLine):
            CleanedText = '\n'.join([line for line in Text.splitlines()])

        SPLIT_PATTERN = re.compile(r'[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+')

        splits = re.compile(words_idicating_similarity).split(CleanedText)

        SplitText = re.findall(SPLIT_PATTERN, CleanedText)
        SplitText = [match for split in splits for match in SPLIT_PATTERN.findall(split)]

        return SplitText

    def TrainVocab(self, FilePath, VocabSize, PrintStat, PrintStatsEvery_Token, WithoutNewLine, SkipFirstChunkInLine, Replacements, RemoveSpecialTok):
        """
        This function is implemented in such a way that     
        it builds upon the existing vocabulary and does 
        not start the training from tha scratch. 
        This is implemented in such a way so that this 
        function can be called multiple times to train 
        a larger vocabulary size as needed by requirements 
        in further control flow.
        """

        if RemoveSpecialTok:
            print('Removing Special Tokens from the Vocabulary End')
            if len(self.Vocab) > 256:
                self.Vocab = self.Vocab[:-len(self.special_tokens)]

        Text = open(FilePath, "r", encoding="utf-8").read()
        return self.TrainVocab_fromText(Text, VocabSize, PrintStat, PrintStatsEvery_Token, WithoutNewLine, SkipFirstChunkInLine, Replacements)
            
    def count_bigrams(self, segment):
        return Counter(zip(segment, segment[1:]))

    def TrainVocab_fromText(self, Text, VocabSize, PrintStat, PrintStatsEvery_Token, WithoutNewLine, SkipFirstChunkInLine, Replacements): 
        
        TextTokens = self.EncodeFromText(Text, WithoutNewLine, SkipFirstChunkInLine, Replacements)

        Token = 0
        if(PrintStat):
            print(f"Vocab Training Started:\nTokens Trained:")
        
        while(len(self.Vocab) < (VocabSize - len(self.special_tokens))):

            Stats = Counter(itertools.chain.from_iterable(zip(seg, seg[1:]) for seg in TextTokens))

            Bigram = max(Stats, key = lambda item: Stats.get(item))
            NewToken = self.Vocab[-1] + 1
            self.MergeInfo[Bigram] = NewToken
            self.Vocab.append(NewToken)

            TextTokens = [self.Merge(seg, Bigram, NewToken) for seg in TextTokens]

            if PrintStat and (Token % PrintStatsEvery_Token == 0): 
                print(len(self.Vocab), end = " - Repetitions of Bigram: ")
                print(Stats[Bigram])

            Token += 1

            if len(self.Vocab) % 1000 == 0:
                self.save('./Tokenizer/', 'Final-Corpus-Tokenizer-Merge-Info-NL-' + str(len(self.Vocab)) + '-', 'Final-Corpus-Tokenizer-Vocab-NL-' + str(len(self.Vocab)) + '-', Save_SpecialTok = False)
                
                self.PrintTokenizedText(TextTokens, SaveFilePath = './Tokenizer/Tokenized-Final-Corpus-' + str(len(self.Vocab)) + '.json')

        return TextTokens, self.Vocab

    def Encode(self, FilePath, WithoutNewLine, SkipFirstChunkInLine, Replacements):
        Text = open(FilePath, "r", encoding="utf-8").read()

        return self.EncodeFromText(Text, WithoutNewLine, SkipFirstChunkInLine, Replacements)

    def DecodeVocab(self, SaveFilePath = None):
        
        DecodedBytes = {}
        
        for i in range(256):
            DecodedBytes[i] = bytes([i])

        for (m0, m1), mgd in self.MergeInfo.items():
            DecodedBytes[mgd] = DecodedBytes[m0] + DecodedBytes[m1]

        TokenizedText = [DecodedBytes[tok].decode("utf-8", errors="replace") for tok in self.Vocab[:-len(self.special_tokens)]]

        if SaveFilePath != None:
            with open(SaveFilePath, 'w', encoding='utf-8') as file:
                json.dump(TokenizedText, file, ensure_ascii=False, indent=4)

                print(f"Vocab saved to: {SaveFilePath}")

    def EncodeFromText(self, Text, WithoutNewLine, SkipFirstChunkInLine, Replacements):

        Text = self.GetCleanedText(Text, WithoutNewLine, SkipFirstChunkInLine, Replacements)

        Encoded = []

        if self.MergeInfo == {}:
            for seg in Text:
                Tokens = list(seg.encode("utf-8"))
                Encoded.append(Tokens)
                print('Vocab is Empty!!')
        else:
            print(f'Encoding from Vocab - (Size: {len(self.Vocab)})')
            for seg in Text:
                Tokens = list(seg.encode("utf-8"))

                while len(Tokens) >= 2:
                    Stats = {}
                    self.GetBigramStats(Tokens, Stats)
                    Bigram = min(Stats, key = lambda p: self.MergeInfo.get(p, float("inf")))

                    if Bigram not in self.MergeInfo:
                        break
                    
                    NewToken = self.MergeInfo[Bigram]
                    Tokens = self.Merge(Tokens, Bigram, NewToken)
                
                Encoded.append(Tokens)
        
        return Encoded

    def Decode(self, Tokens):
        
        DecodedBytes = {}
        
        for i in range(256):
            DecodedBytes[i] = bytes([i])

        for (m0, m1), mgd in self.MergeInfo.items():
            DecodedBytes[mgd] = DecodedBytes[m0] + DecodedBytes[m1]
        
        # These two approaches need to be switch according to the form in which Tokens is passed. Better logic can be implemented here and is a part of the To-Do list of the project.
        Bytes = [b"".join(DecodedBytes[idx] for idx in Tokens)]
        # Bytes = [b"".join(DecodedBytes[idx] for idx in seg) for seg in Tokens]
        
        return "".join(Chunk.decode("utf-8", errors="replace") for Chunk in Bytes)

    def PrintTokenizedText(self, Tokens, SaveFilePath = None):
        
        DecodedBytes = {}
        
        for i in range(256):
            DecodedBytes[i] = bytes([i])

        for (m0, m1), mgd in self.MergeInfo.items():
            DecodedBytes[mgd] = DecodedBytes[m0] + DecodedBytes[m1]
        
        TokenizedText = [[DecodedBytes[idx].decode("utf-8", errors="replace") for idx in seg] for seg in Tokens]

        if SaveFilePath != None:
            with open(SaveFilePath, 'w', encoding='utf-8') as file:
                json.dump(TokenizedText, file, ensure_ascii=False, indent=4)

                print(f"Tokenized text saved to: {SaveFilePath}")

        # print(TokenizedText)
    
    def save(self, path, MergeInfo_Name, Vocab_Name, Save_SpecialTok):
        
        if Save_SpecialTok:
            for i in range(len(self.special_tokens)):
                self.Vocab.append(len(self.Vocab))

        if(path[-1] != '/'): path += '/'
        
        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')
        
        with open(path + MergeInfo_Name + date_time + '.pkl', 'wb') as file:
            pickle.dump(self.MergeInfo, file)

        with open(path + Vocab_Name + date_time + '.pkl', 'wb') as file:
            pickle.dump(self.Vocab, file)

        print("Contents saved successfully.")
        
    def load(self, path, MergeInfo_Name, Vocab_Name):
        
        if(path[-1] != '/'): path += '/'

        with open(path + MergeInfo_Name + '.pkl', 'rb') as file:
            self.MergeInfo = pickle.load(file)
        
        with open(path + Vocab_Name + '.pkl', 'rb') as file:
            self.Vocab = pickle.load(file)

        print('Loaded Vocabulary!')
        print(f'Vocab Size: {len(self.Vocab)}')
        print(f'MergeInfo Size: {len(self.MergeInfo)}')