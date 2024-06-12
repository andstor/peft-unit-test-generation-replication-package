import torch
from transformers import StoppingCriteria




class BraceMatchingCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the number of open brackets is equal to the number of close brackets.
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer used to generate the tokens.
        start_level (:obj:`int`, `optional`, defaults to 0): The initial level of open brackets.
    """

    def __init__(self, tokenizer, start_level=0):
        self.tokenizer = tokenizer
        bracket_tokens = self.find_tokens(tokenizer, "}")
        bracket_tokens.update(self.find_tokens(tokenizer, "{"))
        self.token_indent_value = {id: self.calculate_increase(token) for token, id in bracket_tokens.items()}

        self.start_level = start_level
        self.cnt = start_level
        self.cnt_max = start_level
        
        self.state = {}

    def find_tokens(self, tokenizer, substring):
        return {token: tokenizer.vocab[token] for token in tokenizer.vocab.keys() if substring in token}

    def calculate_increase(self, token):
        # get numer of open brackets
        open_brackets = token.count("{")
        close_brackets = token.count("}")
        return open_brackets - close_brackets

    def reset(self):
        self.state = {}

    def stop_reason(self):
        return "brace_matching"

    def get_sequence_state(self, seq_index: int):
        return self.state.setdefault(seq_index, {
            "cnt": self.start_level,
            "cnt_max": self.start_level,
            "stop": False
        })
    
    def stop_sequence(self, seq_index: int):
        self.get_sequence_state(seq_index)["stop"] = True

    def is_sequence_stopped(self, seq_index: int):
        return self.get_sequence_state(seq_index)["stop"]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for i, seq in enumerate(input_ids):
            seq_state = self.get_sequence_state(i)
            if seq_state["stop"]:
                seq[-1] = self.tokenizer.pad_token_id
            else:
                token = seq[-1].item()
                if token in self.token_indent_value:
                    #print("index:", i, "token:", token, "indent:", self.token_indent_value[token], "cnt:", seq_state["cnt"], "cnt_max:", seq_state["cnt_max"])
                    seq_state["cnt"] += self.token_indent_value[token]
                    seq_state["cnt_max"] = max(seq_state["cnt_max"], seq_state["cnt"])
                if seq_state["cnt"] == 0 and seq_state["cnt_max"] > 0:
                    #print("index:", i, "stopping")
                    self.stop_sequence(i)

        # if all sequences are stopped, return True
        if all([self.get_sequence_state(i)["stop"] for i in range(len(input_ids))]):
            return True
        else:
            return False
