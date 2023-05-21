# CS224N final project

This was my final project for CS 224N at Stanford. The objective was to create a question answering model that found the start and finish index of the answer to a question, within a particular context paragraph. This was done on the SQuAD dataset. To accomplish this, I re-implemented the QANet model (Yu et. al 2017) and ensembled multiple variations of this model to produce reasonably good results (EM: 63.787, F1: 66.646) which got a top 10 score on the class leaderboard
