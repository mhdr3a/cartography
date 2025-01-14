import os

from transformers.data.processors.utils import DataProcessor, InputExample


class SNLIProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def get_labels(self):
        return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[8] #"%s-%s" % (set_type, line[0])
            text_a = line[5]
            text_b = line[6]
            label = line[0]
            if label == "-" or label == "":
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_examples(self, data_file, set_type):
        return self._create_examples(self._read_tsv(data_file), set_type=set_type)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "snli_1.0_train.txt"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "snli_1.0_dev.txt"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "snli_1.0_test.txt"), "test")
