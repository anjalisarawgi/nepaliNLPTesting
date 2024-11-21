from datasets import load_dataset

access_token ="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"
dataset = load_dataset("oscar-corpus/OSCAR-2201",
                        # use_auth_token=True, # required
                        language="ne", 
                        # streaming=True, # optional
                        split="train", 
                        token=access_token
                        )

print("number of examples in the dataset: ", len(dataset))