# NLP Experiments

The following experiments are registered in the NLP library and can be used with `dump_stablehlo.py`:

- `bert/sentence_prediction`
- `bert/sentence_prediction_text`
- `bert/squad`
- `bert/tagging`
- `bert/pretraining`
- `bert/pretraining_dynamic`
- `bert/text_wiki_pretraining`
- `electra/pretraining`
- `wmt_transformer/large`

Use `python -m official.nlp.tools.collect_all_stablehlo` to produce StableHLO files for each of these experiments. Results are written under the `output/` directory by default.
