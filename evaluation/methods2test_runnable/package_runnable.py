from datasets import load_dataset, Dataset, Features, Value
import pandas as pd

keywords = ["assert", "verify", "fail"]

def contains_keyword(text):
    for keyword in keywords:
        if keyword in text.lower():
            return True
    return False

def main():
    # Login using e.g. `huggingface-cli login` to access this dataset
    meta_ds = load_dataset("andstor/methods2test_meta", "test_status", split="test")
    methods2test_ds = load_dataset("andstor/methods2test", 'fm+fc+c+m+f+t+tc', split="test")
    
    # Only keep successful test cases
    runnable_df = meta_ds.to_pandas().set_index("id")
    runnable_df = runnable_df[runnable_df["status"] == "success"]

    def gen_ds():
        for row in methods2test_ds:
            if row["id"] in runnable_df.index:
                if contains_keyword(row["target"]):
                    yield row

    ds = Dataset.from_generator(generator=gen_ds, features=methods2test_ds.features, split="test")
    ds.push_to_hub("andstor/methods2test_runnable", "fm+fc+c+m+f+t+tc", private=False, max_shard_size="250MB")

if __name__ == "__main__":
    main()
