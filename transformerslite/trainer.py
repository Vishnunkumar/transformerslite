from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

class SeqClassifier:

    def __init__(self, dataset, model, tokenizer):

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer