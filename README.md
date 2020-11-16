# corti_text-based_anomaly_detection

Currently we are in the process of transforming into the new architecture not using x in the decoder. In this process the inference.py will probably not work right now. Word dropout is also removed will need to be reimplemented.


I (Marcus) added encoder and decoder funtions feel free to move what you think belongs in which function.

ptb.py and model.py are edited to it the new architecture. The inference function of model.py is not modified for the new architecture. This does not affect that the model should still be able to train and learn. We are just without a function to generate sentences from z.

model.py has some minor modifications to make it work with my cuda.

train.py has in it's evaluation of the loss a minor modification where it was previously able to remove some padding. it does not do that anymore but might be reimplementable later.
As I had problems with the saving files with the names suggested in the code with windows currently to run train.py you need to first make the folder you want it to safe in and then tell it what that folder is. 
