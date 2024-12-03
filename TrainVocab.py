import torch
import time
from BPETokenizer import *

"""
Following is a List of Hyperparameters:
"""
VocabSize = 1024*50
File = "Final-Corpus-Tokenizer.txt"
FilePath = "./"
replacements = {"<br>": "", "<p>": "", "</p>": ""}
VocabPath = "./"
Save_MergeInfo_Name = 'Final-Corpus-Tokenizer-Merge-Info-NL-' + str(VocabSize) + '-'
Save_Vocab_Name = 'Final-Corpus-Tokenizer-Vocab-NL-' + str(VocabSize) + '-'
WithoutNewLine = False
SkipFirstChunkInLine = False
Tokenization_File = './Tokenized-Final-Corpus-' + str(VocabSize) + '.json'
"""
List Ends
"""

StartTime = time.time()

Tokenizer = BPETokenizer()

# Tokenizer.load("./Vocab/", Load_MergeInfo_Name, Load_Vocab_Name)

TextTokens, Vocab = Tokenizer.TrainVocab(
                                            FilePath + File, 
                                            VocabSize, 
                                            PrintStat = True, 
                                            PrintStatsEvery_Token = 1, 
                                            WithoutNewLine = WithoutNewLine, 
                                            SkipFirstChunkInLine = SkipFirstChunkInLine, 
                                            Replacements = replacements
                                        )

Tokenizer.save(VocabPath, Save_MergeInfo_Name, Save_Vocab_Name)

Tokenizer.PrintTokenizedText(TextTokens, SaveFilePath = Tokenization_File)

print(f"Time Taken: {time.time() - StartTime} s")