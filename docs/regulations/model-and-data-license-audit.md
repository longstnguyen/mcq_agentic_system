# Model and Training Data License Audit

This audit is required by the Innovator notice stating that models and training
data must use licenses that permit the intended use.

## Inference Model

| Resource | Status | Required evidence |
| --- | --- | --- |
| `unsloth/Qwen3-1.7B` base model | Apache-2.0 upstream model family | Preserve the upstream license and model card in the final repository/image |
| ViTutor Qwen3-1.7B epoch 3 adapter | Team-created; the packaged model card and repository now declare Apache-2.0 | Mirror the same license metadata to the Hugging Face repository before submission |

The packaged adapter inherits a compatible open base and now has a local license
declaration. The remote Hugging Face card must match the packaged declaration so
reviewers see one consistent license.

## Training Data

The final Docker image contains no training corpus and no retrieval corpus.
Nevertheless, the competition notice applies to the data used to train the
submitted adapter. Before submission, the team must attach the ViTutor data
manifest and verify each source category:

| Source category | Required check |
| --- | --- |
| Team-generated educational interactions | Confirm the team owns the generated records and may release/use them |
| Public reasoning or instruction datasets | Record dataset name, version, URL, and license |
| Translated datasets | Verify that the source license permits derivatives and translation |
| OCR, textbook, and reference material | Verify explicit reuse permission; public accessibility alone is not a license |

Any source without a documented compatible license should be removed from the
training claim or replaced by a properly licensed source. This audit cannot be
completed solely from the files packaged in `mcq_agentic_system`.
